"""Fenêtre « Animer une image (IA) » : image de référence + vidéo pilote → vidéo animée."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets

from ..ai.animate import AnimateOptions
from .workers import AnimateWorker

IMG_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
VID_FILTER = "Vidéos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;Tous les fichiers (*)"


def _fmt(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600:d} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60:d} min {s % 60:02d} s"


class AnimateDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Animer une image (IA)")
        self.setMinimumWidth(560)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[AnimateWorker] = None

        self.ref_edit = QtWidgets.QLineEdit()
        self.drv_edit = QtWidgets.QLineEdit()
        self.out_edit = QtWidgets.QLineEdit()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Visage et tête (IA LivePortrait, meilleure qualité)", "face")
        self.mode_combo.addItem("Corps entier (marionnette, approximatif)", "body")
        form = QtWidgets.QFormLayout()
        form.addRow("Mode", self.mode_combo)
        form.addRow("Image de référence", self._row(self.ref_edit, self._pick_ref))
        form.addRow("Vidéo pilote", self._row(self.drv_edit, self._pick_drv))
        form.addRow("Fichier de sortie", self._row(self.out_edit, self._pick_out))

        self.side_spin = QtWidgets.QSpinBox()
        self.side_spin.setRange(256, 2048)
        self.side_spin.setSingleStep(128)
        self.side_spin.setValue(768)
        self.side_spin.setSuffix(" px")
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(15)
        self.fps_spin.setSuffix(" img/s")
        self.dur_spin = QtWidgets.QDoubleSpinBox()
        self.dur_spin.setRange(0, 3600)
        self.dur_spin.setSpecialValueText("toute la vidéo")
        self.dur_spin.setSuffix(" s")
        self.int_spin = QtWidgets.QDoubleSpinBox()
        self.int_spin.setRange(0.2, 2.0)
        self.int_spin.setSingleStep(0.1)
        self.int_spin.setValue(1.0)
        self.audio_check = QtWidgets.QCheckBox("Conserver le son de la vidéo pilote")
        self.audio_check.setChecked(True)
        form.addRow("Taille de sortie (plus grand côté)", self.side_spin)
        form.addRow("Cadence traitée", self.fps_spin)
        form.addRow("Durée à traiter", self.dur_spin)
        form.addRow("Intensité du mouvement", self.int_spin)
        form.addRow("", self.audio_check)

        note = QtWidgets.QLabel(
            "Visage : le mouvement de la tête et des expressions est copié sur le visage de l'image. "
            "Corps entier : le squelette de la vidéo déforme l'image (rapide mais grossier : fond et "
            "vêtements s'étirent, rien n'est inventé pour les zones cachées). "
            "Tout se calcule sur votre PC (aucun envoi sur Internet) ; seul le premier lancement "
            "télécharge les modèles IA. Sur un processeur seul, comptez plusieurs secondes par image : "
            "testez d'abord avec 2 à 3 secondes de vidéo."
        )
        note.setWordWrap(True)
        self.status = QtWidgets.QLabel("")
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.run_btn = QtWidgets.QPushButton("Lancer")
        self.cancel_btn = QtWidgets.QPushButton("Annuler")
        self.cancel_btn.setEnabled(False)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.cancel_btn)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        lay.addWidget(self.status)
        lay.addWidget(self.bar)
        lay.addLayout(buttons)

        self.run_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)

    # -- aides
    def _row(self, edit: QtWidgets.QLineEdit, slot) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        btn = QtWidgets.QPushButton("Parcourir…")
        btn.clicked.connect(slot)
        h.addWidget(edit, 1)
        h.addWidget(btn)
        return w

    def _pick_ref(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Image de référence", "", IMG_FILTER)
        if p:
            self.ref_edit.setText(p)
            if not self.out_edit.text():
                self.out_edit.setText(str(Path(p).with_name(Path(p).stem + "_anime.mp4")))

    def _pick_drv(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Vidéo pilote", "", VID_FILTER)
        if p:
            self.drv_edit.setText(p)

    def _pick_out(self) -> None:
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Fichier de sortie", self.out_edit.text(), "MP4 (*.mp4)")
        if p:
            self.out_edit.setText(p if p.lower().endswith(".mp4") else p + ".mp4")

    def _set_busy(self, busy: bool) -> None:
        self.run_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    # -- exécution
    def _start(self) -> None:
        ref, drv, out = self.ref_edit.text().strip(), self.drv_edit.text().strip(), self.out_edit.text().strip()
        if not (Path(ref).is_file() and Path(drv).is_file() and out):
            QtWidgets.QMessageBox.warning(self, "Champs manquants", "Choisissez l'image, la vidéo et le fichier de sortie.")
            return
        opts = AnimateOptions(
            reference=ref, driving=drv, output=out, max_output_side=self.side_spin.value(),
            max_fps=self.fps_spin.value(), duration=self.dur_spin.value() or None,
            multiplier=self.int_spin.value(), audio=self.audio_check.isChecked(),
            mode=self.mode_combo.currentData(),
        )
        self._thread = QtCore.QThread(self)
        self._worker = AnimateWorker(opts)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status.connect(self.status.setText)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self.bar.setValue(0)
        self._set_busy(True)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Annulation…")

    def _on_progress(self, done: int, total: int, fps: float, eta: float) -> None:
        self.bar.setValue(int(100 * done / max(1, total)))
        if fps > 0:
            self.status.setText(f"Image {done}/{total} — {fps:.2f} img/s — reste {_fmt(eta)}")

    def _end(self) -> None:
        self._set_busy(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        self._thread = None
        self._worker = None

    def _on_done(self, path: str) -> None:
        self._end()
        self.bar.setValue(100)
        self.status.setText(f"Terminé : {path}")
        QtWidgets.QMessageBox.information(self, "Terminé", f"Vidéo créée :\n{path}")

    def _on_failed(self, msg: str) -> None:
        self._end()
        self.status.setText("Échec.")
        QtWidgets.QMessageBox.critical(self, "Erreur", msg)

    def _on_cancelled(self) -> None:
        self._end()
        self.status.setText("Annulé.")

    def reject(self) -> None:  # fermeture de la fenêtre pendant un calcul
        if self._worker is not None:
            self._worker.cancel()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(8000)
        super().reject()
