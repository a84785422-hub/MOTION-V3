"""Main window of Video Motion Control."""

from __future__ import annotations

import copy
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .. import __version__
from ..camera import CameraState, CameraTrack, Keyframe, view_corners
from ..curves import normalize_interp
from ..export import check_project, render_settings_for
from ..presets import PRESETS
from ..project import Project, ProjectError
from ..render import QUALITIES, Renderer, SourceFrame
from ..video_io import MediaError, MediaInfo, grab_frame, probe
from .widgets import PreviewWidget, TimelineWidget, format_time, to_qimage
from .workers import PREVIEW_FPS, ExportWorker, PreviewWorker

CUSTOM_INTERP = "__custom__"

INTERP_LABELS = [
    ("ease_in_out", "Accélération + décélération"),
    ("ease_in", "Accélération"),
    ("ease_out", "Décélération"),
    ("ease_in_cubic", "Accélération forte"),
    ("ease_out_cubic", "Décélération forte"),
    ("ease_in_out_cubic", "Douce (cubique)"),
    ("ease_in_out_quint", "Très douce (quintique)"),
    ("linear", "Linéaire"),
    ("smooth", "Spline lisse (sans arrêt entre clés)"),
    ("hold", "Maintien (saut)"),
]

SIZE_PRESETS = [
    ("Taille de la source", None),
    ("1280 × 720 (HD)", (1280, 720)),
    ("1920 × 1080 (Full HD)", (1920, 1080)),
    ("2560 × 1440 (2K)", (2560, 1440)),
    ("3840 × 2160 (4K UHD)", (3840, 2160)),
    ("1080 × 1920 (vertical)", (1080, 1920)),
    ("1080 × 1080 (carré)", (1080, 1080)),
]

MEDIA_FILTER = (
    "Médias (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mxf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;"
    "Tous les fichiers (*)"
)


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def _spin(lo: float, hi: float, step: float, decimals: int, value: float = 0.0, suffix: str = "") -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(suffix)
    return box


class MainWindow(QtWidgets.QMainWindow):
    PREVIEW_HEIGHT = 360

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Motion Control")
        self.resize(1560, 940)
        self.setAcceptDrops(True)

        self.project = Project()
        self.project_path: Optional[str] = None
        self.info: Optional[MediaInfo] = None
        self.track = CameraTrack([])
        self.t = 0.0
        self._updating = False
        self._frame_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._last_output: Optional[str] = None

        self._export_thread: Optional[QtCore.QThread] = None
        self._export_worker: Optional[ExportWorker] = None
        self._anim_thread: Optional[QtCore.QThread] = None
        self._anim_worker: Optional[PreviewWorker] = None
        self._anim_frames: list[QtGui.QImage] = []
        self._anim_index = 0
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(45)
        self._preview_timer.timeout.connect(self._update_previews)

        self._build_ui()
        self._build_menu()
        self._set_loaded(False)
        self.statusBar().showMessage("Ouvrez une vidéo ou une image (Ctrl+O) ou glissez-la dans la fenêtre.")

    # ------------------------------------------------------------------ UI construction

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self._build_left_panel())
        root.addLayout(self._build_center(), 1)
        root.addWidget(self._build_right_panel())

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(340)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        group = QtWidgets.QGroupBox("Caméra à l'instant courant")
        form = QtWidgets.QFormLayout(group)
        self.x_spin = _spin(-2.0, 3.0, 0.005, 4, 0.5)
        self.y_spin = _spin(-2.0, 3.0, 0.005, 4, 0.5)
        self.zoom_spin = _spin(0.05, 50.0, 0.01, 3, 1.0)
        self.rot_spin = _spin(-3600.0, 3600.0, 0.1, 2, 0.0, " °")
        self.x_spin.setToolTip("Position horizontale du centre de la caméra (0 = bord gauche, 1 = bord droit).")
        self.y_spin.setToolTip("Position verticale du centre de la caméra (0 = haut, 1 = bas).")
        self.zoom_spin.setToolTip("1 = cadrage de base, 2 = on voit la moitié de l'image.")
        self.rot_spin.setToolTip("Rotation en degrés, positive = sens horaire.")
        for label, box in (("Position X", self.x_spin), ("Position Y", self.y_spin),
                           ("Zoom", self.zoom_spin), ("Rotation", self.rot_spin)):
            form.addRow(label, box)
            box.valueChanged.connect(self._on_camera_edited)
        self.interp_combo = QtWidgets.QComboBox()
        self.interp_combo.addItem("Personnalisée (fichier projet)", CUSTOM_INTERP)
        for key, label in INTERP_LABELS:
            self.interp_combo.addItem(label, key)
        self.interp_combo.setToolTip(
            "Courbe du mouvement entre cette image clé et la suivante.\n"
            "Pour un mouvement continu à travers plusieurs clés, choisissez « Spline lisse »."
        )
        self.interp_combo.currentIndexChanged.connect(self._on_interp_changed)
        form.addRow("Courbe →", self.interp_combo)
        lay.addWidget(group)

        row = QtWidgets.QHBoxLayout()
        self.add_key_btn = QtWidgets.QPushButton("Image clé ici")
        self.add_key_btn.setToolTip("Crée ou met à jour une image clé à l'instant courant.")
        self.add_key_btn.clicked.connect(self._on_add_key_clicked)
        self.del_key_btn = QtWidgets.QPushButton("Supprimer")
        self.del_key_btn.clicked.connect(self._on_delete_key_clicked)
        row.addWidget(self.add_key_btn)
        row.addWidget(self.del_key_btn)
        lay.addLayout(row)

        keys_group = QtWidgets.QGroupBox("Images clés")
        kl = QtWidgets.QVBoxLayout(keys_group)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["t (s)", "X", "Y", "Zoom", "Rot.", "Courbe"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        kl.addWidget(self.table)
        lay.addWidget(keys_group, 1)
        return panel

    def _build_center(self) -> QtWidgets.QVBoxLayout:
        lay = QtWidgets.QVBoxLayout()
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.src_view = PreviewWidget("Source + cadrage de la caméra")
        self.out_view = PreviewWidget("Rendu (aperçu rapide)")
        split.addWidget(self.src_view)
        split.addWidget(self.out_view)
        lay.addWidget(split, 1)

        self.timeline = TimelineWidget()
        self.timeline.seek_requested.connect(self._on_timeline_seek)
        lay.addWidget(self.timeline)

        row = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀ Image")
        self.next_btn = QtWidgets.QPushButton("Image ▶")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(1))
        self.time_spin = _spin(0.0, 36000.0, 0.04, 3, 0.0, " s")
        self.time_spin.valueChanged.connect(self._on_time_spin)
        self.time_label = QtWidgets.QLabel("")
        self.play_btn = QtWidgets.QPushButton("▶ Aperçu animé")
        self.play_btn.setToolTip("Calcule un aperçu basse résolution du mouvement complet (avec flou) puis le lit en boucle.")
        self.play_btn.clicked.connect(self._on_play_clicked)
        self.anim_progress = QtWidgets.QProgressBar()
        self.anim_progress.setFixedWidth(140)
        self.anim_progress.setVisible(False)
        for w in (self.prev_btn, self.time_spin, self.next_btn):
            row.addWidget(w)
        row.addWidget(self.time_label, 1)
        row.addWidget(self.anim_progress)
        row.addWidget(self.play_btn)
        lay.addLayout(row)
        return lay

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(350)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        src = QtWidgets.QGroupBox("Plage de la source")
        sf = QtWidgets.QFormLayout(src)
        self.start_spin = _spin(0.0, 36000.0, 0.1, 3, 0.0, " s")
        self.end_spin = _spin(0.0, 36000.0, 0.1, 3, 0.0, " s")
        self.end_spin.setSpecialValueText("jusqu'à la fin")
        self.still_spin = _spin(0.1, 3600.0, 0.5, 2, 5.0, " s")
        self.still_label = QtWidgets.QLabel("Durée (image fixe)")
        sf.addRow("Début", self.start_spin)
        sf.addRow("Fin", self.end_spin)
        sf.addRow(self.still_label, self.still_spin)
        lay.addWidget(src)

        out = QtWidgets.QGroupBox("Sortie")
        of = QtWidgets.QFormLayout(out)
        self.preset_combo = QtWidgets.QComboBox()
        for key, preset in PRESETS.items():
            self.preset_combo.addItem(preset.label, key)
        self.size_combo = QtWidgets.QComboBox()
        for label, _ in SIZE_PRESETS:
            self.size_combo.addItem(label)
        self.size_combo.currentIndexChanged.connect(self._on_size_preset)
        self.width_spin = QtWidgets.QSpinBox()
        self.height_spin = QtWidgets.QSpinBox()
        for s in (self.width_spin, self.height_spin):
            s.setRange(16, 16384)
            s.setSingleStep(2)
            s.setKeyboardTracking(False)
        self.fps_spin = _spin(1.0, 480.0, 1.0, 3, 30.0, " i/s")
        self.fit_combo = QtWidgets.QComboBox()
        self.fit_combo.addItem("Remplir le cadre (recadrer)", "cover")
        self.fit_combo.addItem("Image entière (bandes noires)", "contain")
        self.quality_combo = QtWidgets.QComboBox()
        for key, label in (("draft", "Brouillon (rapide)"), ("high", "Haute"), ("ultra", "Ultra (très lent)")):
            self.quality_combo.addItem(label, key)
        self.quality_combo.setCurrentIndex(1)
        sz = QtWidgets.QHBoxLayout()
        sz.addWidget(self.width_spin)
        sz.addWidget(QtWidgets.QLabel("×"))
        sz.addWidget(self.height_spin)
        of.addRow("Format", self.preset_combo)
        of.addRow("Taille", self.size_combo)
        of.addRow("", sz)
        of.addRow("Cadence", self.fps_spin)
        of.addRow("Cadrage", self.fit_combo)
        of.addRow("Qualité", self.quality_combo)
        lay.addWidget(out)

        fx = QtWidgets.QGroupBox("Flou de mouvement")
        ff = QtWidgets.QFormLayout(fx)
        self.blur_check = QtWidgets.QCheckBox("Activer")
        self.blur_check.setChecked(True)
        self.shutter_spin = _spin(0.0, 360.0, 10.0, 0, 180.0, " °")
        self.shutter_spin.setToolTip("Angle d'obturation : 180° = rendu cinéma classique, 360° = flou maximal.")
        self.linear_check = QtWidgets.QCheckBox("Mélange en lumière linéaire (plus réaliste)")
        self.linear_check.setChecked(True)
        self.dither_check = QtWidgets.QCheckBox("Tramage anti-banding")
        self.dither_check.setChecked(True)
        self.audio_check = QtWidgets.QCheckBox("Conserver l'audio")
        self.audio_check.setChecked(True)
        ff.addRow(self.blur_check)
        ff.addRow("Obturateur", self.shutter_spin)
        ff.addRow(self.linear_check)
        ff.addRow(self.dither_check)
        ff.addRow(self.audio_check)
        lay.addWidget(fx)

        path_row = QtWidgets.QHBoxLayout()
        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText("Fichier de sortie…")
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(34)
        browse.clicked.connect(self._on_browse_output)
        path_row.addWidget(self.out_edit, 1)
        path_row.addWidget(browse)
        lay.addLayout(path_row)

        btns = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Exporter")
        self.export_btn.setDefault(True)
        self.export_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.cancel_btn = QtWidgets.QPushButton("Annuler")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_export)
        btns.addWidget(self.export_btn, 1)
        btns.addWidget(self.cancel_btn)
        lay.addLayout(btns)

        self.export_progress = QtWidgets.QProgressBar()
        self.export_status = QtWidgets.QLabel("")
        self.export_status.setWordWrap(True)
        self.reveal_btn = QtWidgets.QPushButton("Afficher le fichier exporté")
        self.reveal_btn.setVisible(False)
        self.reveal_btn.clicked.connect(self._on_reveal)
        lay.addWidget(self.export_progress)
        lay.addWidget(self.export_status)
        lay.addWidget(self.reveal_btn)
        lay.addStretch(1)

        # Settings that change what the preview shows.
        for sig in (
            self.width_spin.valueChanged, self.height_spin.valueChanged, self.fps_spin.valueChanged,
            self.fit_combo.currentIndexChanged, self.blur_check.toggled, self.shutter_spin.valueChanged,
            self.linear_check.toggled, self.start_spin.valueChanged, self.end_spin.valueChanged,
            self.still_spin.valueChanged,
        ):
            sig.connect(self._on_settings_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        return panel

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&Fichier")

        def add(menu, text, slot, shortcut=None):
            act = QtGui.QAction(text, self)
            if shortcut is not None:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            menu.addAction(act)
            return act

        add(file_menu, "Ouvrir un média…", self._on_open_media, QtGui.QKeySequence.StandardKey.Open)
        add(file_menu, "Ouvrir un projet…", self._on_open_project)
        file_menu.addSeparator()
        self.save_action = add(file_menu, "Enregistrer le projet", self._on_save_project, QtGui.QKeySequence.StandardKey.Save)
        self.save_as_action = add(file_menu, "Enregistrer le projet sous…", self._on_save_project_as)
        file_menu.addSeparator()
        add(file_menu, "Animer une image (IA)…", self._on_animate_ai)
        file_menu.addSeparator()
        add(file_menu, "Quitter", self.close, QtGui.QKeySequence.StandardKey.Quit)
        help_menu = bar.addMenu("&Aide")
        add(help_menu, "À propos", self._on_about)

    # ------------------------------------------------------------------ state helpers

    def _on_animate_ai(self) -> None:
        from .animate_dialog import AnimateDialog

        AnimateDialog(self).exec()

    def _set_loaded(self, loaded: bool) -> None:
        for w in (
            self.x_spin, self.y_spin, self.zoom_spin, self.rot_spin, self.interp_combo,
            self.add_key_btn, self.del_key_btn, self.prev_btn, self.next_btn, self.time_spin,
            self.play_btn, self.export_btn, self.table, self.save_action, self.save_as_action,
        ):
            w.setEnabled(loaded)

    def _duration(self) -> float:
        if self.info is None:
            return 0.0
        if self.info.is_image:
            return self.project.still_duration
        end = self.project.end if self.project.end is not None else (self.info.duration or 0.0)
        if self.info.duration:
            end = min(end, self.info.duration)
        return max(0.0, end - self.project.start)

    def _fps(self) -> float:
        fps = self.project.output.fps
        if fps:
            return float(fps)
        if self.info and self.info.fps:
            return float(self.info.fps)
        return 30.0

    def _out_size(self) -> tuple[int, int]:
        return _even(self.width_spin.value()), _even(self.height_spin.value())

    def _rebuild_track(self) -> None:
        try:
            self.track = CameraTrack(self.project.keyframes)
        except ValueError as exc:
            self.statusBar().showMessage(f"Images clés invalides : {exc}")

    def _key_index_at(self, t: float) -> Optional[int]:
        tol = 0.5 / max(self._fps(), 1.0)
        best, best_d = None, tol
        for i, k in enumerate(self.project.keyframes):
            d = abs(k.t - t)
            if d <= best_d:
                best, best_d = i, d
        return best

    # ------------------------------------------------------------------ settings <-> project

    def _push_settings(self) -> None:
        """Project -> widgets."""
        p, info = self.project, self.info
        assert info is not None
        self._updating = True
        try:
            self.preset_combo.setCurrentIndex(max(0, self.preset_combo.findData(p.output.preset)))
            self.width_spin.setValue(p.output.width or _even(info.width))
            self.height_spin.setValue(p.output.height or _even(info.height))
            self.size_combo.setCurrentIndex(0)
            self.fps_spin.setValue(p.output.fps or (float(info.fps) if info.fps else 30.0))
            self.fit_combo.setCurrentIndex(max(0, self.fit_combo.findData(p.output.fit)))
            self.quality_combo.setCurrentIndex(max(0, self.quality_combo.findData(p.output.quality)))
            self.blur_check.setChecked(p.motion_blur.enabled)
            self.shutter_spin.setValue(p.motion_blur.shutter_angle)
            self.linear_check.setChecked(p.linear_light)
            self.dither_check.setChecked(p.dither)
            self.audio_check.setChecked(p.audio and info.has_audio)
            self.audio_check.setEnabled(info.has_audio)
            self.start_spin.setValue(p.start)
            self.end_spin.setValue(p.end or 0.0)
            self.still_spin.setValue(p.still_duration)
            for w in (self.start_spin, self.end_spin, self.still_spin):
                w.setVisible(True)
            self.start_spin.setEnabled(not info.is_image)
            self.end_spin.setEnabled(not info.is_image)
            self.still_spin.setVisible(info.is_image)
            self.still_label.setVisible(info.is_image)
            self.out_edit.setText(p.output.path)
        finally:
            self._updating = False

    def _collect_settings(self) -> None:
        """Widgets -> project."""
        p = self.project
        p.output.preset = self.preset_combo.currentData()
        p.output.width, p.output.height = self._out_size()
        p.output.fps = float(self.fps_spin.value())
        p.output.fit = self.fit_combo.currentData()
        p.output.quality = self.quality_combo.currentData()
        p.motion_blur.enabled = self.blur_check.isChecked()
        p.motion_blur.shutter_angle = float(self.shutter_spin.value())
        p.linear_light = self.linear_check.isChecked()
        p.dither = self.dither_check.isChecked()
        p.audio = self.audio_check.isChecked()
        p.start = float(self.start_spin.value())
        end = float(self.end_spin.value())
        p.end = end if end > p.start else None
        p.still_duration = float(self.still_spin.value())
        p.output.path = self.out_edit.text().strip()

    # ------------------------------------------------------------------ opening files

    def _on_open_media(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Ouvrir un média", "", MEDIA_FILTER)
        if path:
            self.open_media(path)

    def _on_open_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Ouvrir un projet", "", "Projet (*.json);;Tous (*)")
        if path:
            self.open_project(path)

    def open_media(self, path: str) -> bool:
        try:
            info = probe(path)
        except MediaError as exc:
            QtWidgets.QMessageBox.critical(self, "Média illisible", str(exc))
            return False
        self._stop_animation()
        self.info = info
        self.project = Project(source=str(Path(path).resolve()))
        self.project_path = None
        self.project.output.fps = float(info.fps) if info.fps else 30.0
        duration = info.duration if info.duration else self.project.still_duration
        self.project.keyframes = [Keyframe(0.0), Keyframe(round(duration, 3))]
        ext = PRESETS[self.project.output.preset].extension
        self.project.output.path = str(Path(path).with_name(Path(path).stem + "_motion" + ext))
        self._after_load()
        return True

    def open_project(self, path: str) -> bool:
        try:
            project = Project.load(path)
            info = probe(project.source)
        except (ProjectError, MediaError) as exc:
            QtWidgets.QMessageBox.critical(self, "Projet illisible", str(exc))
            return False
        self._stop_animation()
        self.project, self.info, self.project_path = project, info, path
        self._after_load()
        return True

    def _after_load(self) -> None:
        self._frame_cache.clear()
        self._rebuild_track()
        self._push_settings()
        self._set_loaded(True)
        self.timeline.set_duration(self._duration())
        self.time_spin.setMaximum(max(self._duration(), 0.001))
        self._refresh_table()
        self._seek(0.0)
        self.setWindowTitle(f"Video Motion Control : {Path(self.project.source).name}")
        self.statusBar().showMessage(
            f"{self.info.width}×{self.info.height}"
            + (f" · {float(self.info.fps):.3f} i/s" if self.info and self.info.fps else " · image fixe")
        )

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802 - Qt API
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path.lower().endswith(".json"):
            self.open_project(path)
        elif path:
            self.open_media(path)

    # ------------------------------------------------------------------ project save

    def _on_save_project(self) -> None:
        if not self.project_path:
            self._on_save_project_as()
            return
        self._save_to(self.project_path)

    def _on_save_project_as(self) -> None:
        default = str(Path(self.project.source).with_suffix(".vmc.json")) if self.project.source else "projet.vmc.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Enregistrer le projet", default, "Projet (*.json)")
        if path:
            self._save_to(path)

    def _save_to(self, path: str) -> None:
        self._collect_settings()
        try:
            self.project.save(path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Enregistrement impossible", str(exc))
            return
        self.project_path = path
        self.statusBar().showMessage(f"Projet enregistré : {path}", 5000)

    # ------------------------------------------------------------------ seeking & preview

    def _on_timeline_seek(self, t: float) -> None:
        self._stop_animation()
        self._seek(t)

    def _on_time_spin(self, value: float) -> None:
        if self._updating:
            return
        self._stop_animation()
        self._seek(value)

    def _step(self, direction: int) -> None:
        self._stop_animation()
        self._seek(self.t + direction / max(self._fps(), 1.0))

    def _seek(self, t: float) -> None:
        if self.info is None:
            return
        self.t = min(max(t, 0.0), self._duration())
        self.timeline.set_time(self.t)
        self._updating = True
        try:
            self.time_spin.setValue(self.t)
        finally:
            self._updating = False
        self._sync_controls()
        self._update_time_label()
        self._schedule_preview()

    def _update_time_label(self) -> None:
        frame = int(round(self.t * self._fps()))
        self.time_label.setText(f"{format_time(self.t)} / {format_time(self._duration())} · image {frame}")

    def _sync_controls(self) -> None:
        state = self.track.sample(self.t)
        idx = self._key_index_at(self.t)
        self._updating = True
        try:
            self.x_spin.setValue(state.x)
            self.y_spin.setValue(state.y)
            self.zoom_spin.setValue(state.zoom)
            self.rot_spin.setValue(state.rotation)
            self._select_interp_for(self.t)
            self.table.blockSignals(True)
            self.table.clearSelection()
            if idx is not None:
                self.table.selectRow(idx)
            self.table.blockSignals(False)
        finally:
            self._updating = False

    def _segment_key_index(self, t: float) -> Optional[int]:
        """Index of the keyframe starting the segment that contains ``t``."""
        idx = self._key_index_at(t)
        if idx is not None:
            return idx
        before = [i for i, k in enumerate(self.project.keyframes) if k.t <= t]
        return before[-1] if before else None

    def _select_interp_for(self, t: float) -> None:
        idx = self._segment_key_index(t)
        if idx is None:
            return
        interp = self.project.keyframes[idx].interp
        key = interp if isinstance(interp, str) else CUSTOM_INTERP
        self.interp_combo.setCurrentIndex(max(0, self.interp_combo.findData(key)))

    def _schedule_preview(self) -> None:
        self._preview_timer.start()

    def _get_source_frame(self, abs_t: float) -> np.ndarray:
        assert self.info is not None
        fps = float(self.info.fps) if self.info.fps else 30.0
        key = (0,) if self.info.is_image else (int(round(abs_t * fps)),)
        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)
            return cached
        snapped = 0.0 if self.info.is_image else key[0] / fps
        frame = grab_frame(self.info, snapped, max_height=self.PREVIEW_HEIGHT * 2)
        self._frame_cache[key] = frame
        while len(self._frame_cache) > 24:
            self._frame_cache.popitem(last=False)
        return frame

    def _update_previews(self) -> None:
        if self.info is None or self._anim_timer.isActive():
            return
        try:
            frame = self._get_source_frame(self.project.start + self.t)
        except MediaError as exc:
            self.statusBar().showMessage(str(exc))
            return
        ow, oh = self._out_size()
        fit = self.fit_combo.currentData()
        state = self.track.sample(self.t)
        self.src_view.set_image(to_qimage(frame))
        self.src_view.set_overlay(view_corners(state, self.info.size, (ow, oh), fit))

        ph = min(self.PREVIEW_HEIGHT, oh)
        pw = _even(ph * ow / oh)
        self._collect_settings()
        settings = render_settings_for(self.project, 8, quality="draft")
        settings.dither = False
        renderer = Renderer(pw, ph, settings)
        out = renderer.render(SourceFrame(frame, settings.linear_light), self.track, self.t, 1.0 / self._fps())
        self.out_view.set_image(to_qimage(out))
        title = "Rendu (aperçu rapide)"
        if renderer.last_sample_count > 1:
            title += " · flou " + str(renderer.last_sample_count) + " échantillons"
        self.out_view.set_title(title)

    # ------------------------------------------------------------------ keyframe editing

    def _current_state_from_widgets(self) -> CameraState:
        return CameraState(self.x_spin.value(), self.y_spin.value(), self.zoom_spin.value(), self.rot_spin.value())

    def _interp_from_combo(self, fallback: object = "ease_in_out"):
        data = self.interp_combo.currentData()
        return fallback if data in (None, CUSTOM_INTERP) else data

    def _on_camera_edited(self) -> None:
        if self._updating or self.info is None:
            return
        self._upsert_keyframe(self.t, self._current_state_from_widgets())

    def _on_add_key_clicked(self) -> None:
        if self.info is not None:
            self._upsert_keyframe(self.t, self._current_state_from_widgets())

    def _upsert_keyframe(self, t: float, state: CameraState) -> None:
        idx = self._key_index_at(t)
        if idx is None:
            seg = self._segment_key_index(t)
            inherit = self.project.keyframes[seg].interp if seg is not None else "ease_in_out"
            self.project.keyframes.append(
                Keyframe(round(t, 6), state.x, state.y, state.zoom, state.rotation, inherit)
            )
            self.project.keyframes.sort(key=lambda k: k.t)
        else:
            k = self.project.keyframes[idx]
            k.x, k.y, k.zoom, k.rotation = state.x, state.y, state.zoom, state.rotation
        self._rebuild_track()
        self._refresh_table()
        self._sync_controls()
        self._schedule_preview()

    def _on_delete_key_clicked(self) -> None:
        idx = self._key_index_at(self.t)
        if idx is None:
            rows = self.table.selectionModel().selectedRows()
            idx = rows[0].row() if rows else None
        if idx is None:
            self.statusBar().showMessage("Placez la tête de lecture sur une image clé pour la supprimer.", 4000)
            return
        del self.project.keyframes[idx]
        self._rebuild_track()
        self._refresh_table()
        self._sync_controls()
        self._schedule_preview()

    def _on_interp_changed(self) -> None:
        if self._updating or self.info is None:
            return
        data = self.interp_combo.currentData()
        idx = self._segment_key_index(self.t)
        if data in (None, CUSTOM_INTERP) or idx is None:
            return
        self.project.keyframes[idx].interp = normalize_interp(data)
        self._rebuild_track()
        self._refresh_table()
        self._schedule_preview()

    def _refresh_table(self) -> None:
        keys = sorted(self.project.keyframes, key=lambda k: k.t)
        self.project.keyframes = keys
        self.table.blockSignals(True)
        self.table.setRowCount(len(keys))
        for row, k in enumerate(keys):
            interp = k.interp if isinstance(k.interp, str) else "perso."
            for col, text in enumerate(
                (f"{k.t:.3f}", f"{k.x:.3f}", f"{k.y:.3f}", f"{k.zoom:.3f}", f"{k.rotation:.1f}", interp)
            ):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        self.table.blockSignals(False)
        self.timeline.set_keys([k.t for k in keys])

    def _on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and self.info is not None:
            self._stop_animation()
            self._seek(self.project.keyframes[rows[0].row()].t)

    # ------------------------------------------------------------------ settings changes

    def _on_settings_changed(self) -> None:
        if self._updating or self.info is None:
            return
        self._collect_settings()
        self.timeline.set_duration(self._duration())
        self.time_spin.setMaximum(max(self._duration(), 0.001))
        self._seek(min(self.t, self._duration()))

    def _on_preset_changed(self) -> None:
        if self._updating or self.info is None:
            return
        ext = PRESETS[self.preset_combo.currentData()].extension
        text = self.out_edit.text().strip()
        if text:
            self.out_edit.setText(str(Path(text).with_suffix(ext)))

    def _on_size_preset(self) -> None:
        if self._updating or self.info is None:
            return
        size = SIZE_PRESETS[self.size_combo.currentIndex()][1] or (_even(self.info.width), _even(self.info.height))
        self._updating = True
        try:
            self.width_spin.setValue(size[0])
            self.height_spin.setValue(size[1])
        finally:
            self._updating = False
        self._on_settings_changed()

    def _on_browse_output(self) -> None:
        preset = PRESETS[self.preset_combo.currentData()]
        start = self.out_edit.text().strip() or str(Path(self.project.source).with_suffix(preset.extension))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Fichier de sortie", start, f"Vidéo (*{preset.extension})")
        if path:
            self.out_edit.setText(str(Path(path).with_suffix(preset.extension)))

    # ------------------------------------------------------------------ animated preview

    def _on_play_clicked(self) -> None:
        if self._anim_timer.isActive() or self._anim_thread is not None:
            self._stop_animation()
            return
        if self.info is None:
            return
        self._collect_settings()
        self.play_btn.setText("■ Arrêter")
        self.anim_progress.setValue(0)
        self.anim_progress.setVisible(True)
        thread = QtCore.QThread(self)
        worker = PreviewWorker(copy.deepcopy(self.project), self.info, self.PREVIEW_HEIGHT)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_anim_progress)
        worker.ready.connect(self._on_anim_ready)
        worker.failed.connect(self._on_anim_failed)
        for sig in (worker.ready, worker.failed, worker.cancelled):
            sig.connect(thread.quit)
        thread.finished.connect(self._on_anim_thread_finished)
        self._anim_thread, self._anim_worker = thread, worker
        thread.start()

    def _on_anim_progress(self, done: int, total: int) -> None:
        self.anim_progress.setMaximum(total)
        self.anim_progress.setValue(done)

    def _on_anim_ready(self, frames: list, fps: float) -> None:
        self._anim_frames = frames
        self._anim_index = 0
        self.anim_progress.setVisible(False)
        if frames:
            self._anim_timer.start(int(1000 / fps))

    def _on_anim_failed(self, message: str) -> None:
        self.anim_progress.setVisible(False)
        self.play_btn.setText("▶ Aperçu animé")
        QtWidgets.QMessageBox.warning(self, "Aperçu impossible", message)

    def _on_anim_thread_finished(self) -> None:
        if self._anim_thread is not None:
            self._anim_thread.deleteLater()
        self._anim_thread = None
        self._anim_worker = None

    def _anim_tick(self) -> None:
        if not self._anim_frames:
            return
        self.out_view.set_image(self._anim_frames[self._anim_index])
        self.timeline.set_time(self._anim_index / PREVIEW_FPS)
        self._anim_index = (self._anim_index + 1) % len(self._anim_frames)

    def _stop_animation(self) -> None:
        was_active = self._anim_timer.isActive() or self._anim_thread is not None
        self._anim_timer.stop()
        if self._anim_worker is not None:
            self._anim_worker.cancel()
        if self._anim_thread is not None:
            self._anim_thread.quit()
            self._anim_thread.wait(3000)
        self._anim_frames = []
        self.anim_progress.setVisible(False)
        self.play_btn.setText("▶ Aperçu animé")
        if was_active and self.info is not None:
            self.timeline.set_time(self.t)
            self._schedule_preview()

    # ------------------------------------------------------------------ export

    def _on_export_clicked(self) -> None:
        if self.info is None or self._export_thread is not None:
            return
        self._stop_animation()
        self._collect_settings()
        if not self.project.output.path:
            self._on_browse_output()
            self._collect_settings()
            if not self.project.output.path:
                return
        preset = PRESETS[self.project.output.preset]
        out_path = str(Path(self.project.output.path).with_suffix(preset.extension))
        self.project.output.path = out_path
        self.out_edit.setText(out_path)
        if os.path.exists(out_path):
            answer = QtWidgets.QMessageBox.question(
                self, "Fichier existant", f"{Path(out_path).name} existe déjà. Le remplacer ?"
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        warnings = check_project(self.project, self.info)
        if warnings:
            answer = QtWidgets.QMessageBox.question(
                self, "Vérification du projet", "\n\n".join(warnings) + "\n\nExporter quand même ?"
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self._set_exporting(True)
        thread = QtCore.QThread(self)
        worker = ExportWorker(copy.deepcopy(self.project))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_done)
        worker.failed.connect(self._on_export_failed)
        worker.cancelled.connect(self._on_export_cancelled)
        for sig in (worker.finished, worker.failed, worker.cancelled):
            sig.connect(thread.quit)
        thread.finished.connect(self._on_export_thread_finished)
        self._export_thread, self._export_worker = thread, worker
        thread.start()

    def _set_exporting(self, exporting: bool) -> None:
        self.export_btn.setEnabled(not exporting)
        self.cancel_btn.setEnabled(exporting)
        self.reveal_btn.setVisible(False)
        self.export_progress.setValue(0)
        if exporting:
            self.export_status.setText("Export en cours…")

    def _on_export_progress(self, done: int, total: int, fps: float, eta: float) -> None:
        self.export_progress.setMaximum(total)
        self.export_progress.setValue(done)
        self.export_status.setText(f"Image {done}/{total} · {fps:.1f} i/s · reste {format_time(eta)}")

    def _on_export_done(self, path: str) -> None:
        self._last_output = path
        self.export_progress.setValue(self.export_progress.maximum())
        self.export_status.setText(f"Terminé : {path}")
        self.reveal_btn.setVisible(True)

    def _on_export_failed(self, message: str) -> None:
        self.export_status.setText("L'export a échoué.")
        QtWidgets.QMessageBox.critical(self, "Échec de l'export", message)

    def _on_export_cancelled(self) -> None:
        self.export_status.setText("Export annulé.")

    def _on_export_thread_finished(self) -> None:
        if self._export_thread is not None:
            self._export_thread.deleteLater()
        self._export_thread = None
        self._export_worker = None
        self.export_btn.setEnabled(self.info is not None)
        self.cancel_btn.setEnabled(False)

    def _on_cancel_export(self) -> None:
        if self._export_worker is not None:
            self._export_worker.cancel()
            self.export_status.setText("Annulation…")

    def _on_reveal(self) -> None:
        if self._last_output:
            folder = str(Path(self._last_output).parent)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    # ------------------------------------------------------------------ misc

    def _on_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "À propos",
            f"<b>Video Motion Control {__version__}</b><br>"
            "Mouvements de caméra virtuels (pan, zoom, rotation) avec flou de mouvement "
            "en lumière linéaire et export haute qualité.<br><br>"
            "Utilise FFmpeg pour le décodage et l'encodage.",
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._stop_animation()
        if self._export_worker is not None:
            self._export_worker.cancel()
        if self._export_thread is not None:
            self._export_thread.quit()
            self._export_thread.wait(8000)
        super().closeEvent(event)

    # ------------------------------------------------------------------ self test (CI)

    def selftest(self, media: str) -> str:
        """Exercise the GUI code paths without user input. Returns an error text or ''."""
        if not self.open_media(media):
            return "open_media failed"
        self._seek(self._duration() / 2)
        self._update_previews()
        if self.out_view._image is None or self.src_view._image is None:
            return "preview images were not produced"
        self.zoom_spin.setValue(1.4)  # creates a keyframe through the normal edit path
        if len(self.project.keyframes) < 3:
            return "editing the camera did not create a keyframe"
        self._update_previews()
        return ""
