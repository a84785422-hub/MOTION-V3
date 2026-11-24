# Video Motion Control

## Animer une image avec une vidéo (IA, 100 % locale)

**Fichier → Animer une image (IA)…** : choisissez une **image de référence** (un visage) et une
**vidéo pilote** ; les mouvements de la tête et les expressions de la vidéo sont reproduits sur
l'image, et le son de la vidéo est conservé. Équivalent en ligne de commande :

```powershell
vmc animate visage.png video.mp4 -o resultat.mp4
vmc animate visage.png video.mp4 --duration 3 --max-side 512 --max-fps 12   # test rapide
```

* Moteur : **LivePortrait** (ONNX Runtime, **processeur uniquement**, pas de carte NVIDIA requise).
* **Aucun envoi sur Internet** : seul le premier lancement télécharge les modèles (plusieurs centaines de Mo) dans `%LOCALAPPDATA%\VideoMotionControl\models`. Hors ligne :
  `vmc animate --models-from DOSSIER_AVEC_LES_ONNX --download-models`, ou copiez les `.onnx`
  (`retinaface_det_static`, `face_2dpose_106_static`, `landmark`, `appearance_feature_extractor`,
  `motion_extractor`, `warping_spade`, `stitching`, `stitching_lip`) dans ce dossier.
  Variable `VMC_MODELS` pour changer d'emplacement.
* **Limites** : c'est du transfert de **visage / tête** (pas du corps entier). Le visage de la
  référence doit être net et de face ou presque ; un seul visage est animé (le plus grand).
  La première image de la vidéo pilote sert de point de départ : commencez par un visage neutre.
* **Vitesse** : sur un petit processeur sans GPU, comptez plusieurs secondes par image. Testez avec
  2 à 3 secondes, une sortie de 512 px et 10 à 12 img/s, puis augmentez. Fermez les autres programmes
  (peu de RAM).

---

Mouvements de caméra virtuels (**pan, zoom, rotation**) sur une vidéo ou une image fixe, avec un
**flou de mouvement physique** et un export haute qualité (H.264, H.265 10 bits, ProRes, FFV1,
NVENC). Application Windows x64 (`.exe`) avec interface graphique et outil en ligne de commande.

## Obtenir le .exe (sans rien installer)

Le dépôt contient un workflow GitHub Actions qui compile tout sur un serveur Windows x64.

1. Créez un dépôt GitHub et poussez-y le contenu de ce dossier (ou glissez les fichiers sur la page
   du dépôt).
2. Ouvrez l'onglet **Actions**, workflow **Build Windows x64**. Il se lance à chaque push.
   Vous pouvez aussi le lancer à la main (**Run workflow**).
3. Quand il est vert, ouvrez l'exécution et téléchargez l'artefact
   **VideoMotionControl-windows-x64** (un zip).
4. Décompressez le dossier entier et lancez `VideoMotionControl.exe`.

Pour publier une version téléchargeable depuis la page **Releases** :

```bash
git tag v0.1.0
git push origin v0.1.0
```

Le workflow crée alors la release et y attache le zip.

Le build bloque si les tests échouent, vérifie que les `.exe` sont bien en x64, fait un vrai rendu de
test avec le FFmpeg embarqué, puis ouvre l'interface en mode hors écran pour vérifier qu'elle démarre.

### Compiler en local (Windows 10/11 x64)

Prérequis : Python 3.10 à 3.13 **64 bits**.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

Résultat : `dist\VideoMotionControl\` (`VideoMotionControl.exe` et `vmc.exe`) et
`dist\VideoMotionControl-windows-x64.zip`. Le script télécharge FFmpeg dans `vendor\ffmpeg`.

> Le dossier est volontairement en « one-folder » : démarrage plus rapide et moins de faux positifs
> antivirus qu'un exe unique. Distribuez le dossier complet (ou le zip).

## Utilisation de l'interface

1. **Fichier → Ouvrir un média** (ou glissez une vidéo/image dans la fenêtre).
2. Placez la tête de lecture sur la timeline, puis réglez **Position X/Y**, **Zoom** et **Rotation** :
   chaque modification crée ou met à jour une **image clé** à cet instant. La vue de gauche montre le
   cadre de la caméra sur la source, celle de droite le rendu.
3. Choisissez la **courbe** de chaque segment : accélération/décélération, linéaire, maintien, ou
   **Spline lisse** pour un mouvement continu à travers plusieurs clés (sans arrêt à chaque clé).
4. **Aperçu animé** calcule une version basse résolution du mouvement complet, flou compris, et la lit
   en boucle.
5. Réglez le format, la taille, la cadence, le flou (angle d'obturation 180° = look cinéma), la
   qualité, puis **Exporter**. L'audio de la source est conservé.

Un avertissement s'affiche si la vue sort de l'image source (les bords seraient étirés).
Enregistrez votre travail avec **Fichier → Enregistrer le projet** (fichier `.json`).

## Ligne de commande

```powershell
vmc init clip.mp4 -o projet.json --move push --amount 1.25   # projet de départ
vmc render projet.json --quality ultra -o sortie.mov --preset prores_422hq
vmc probe clip.mp4
vmc presets
```

Mouvements de départ pour `init` : `push`, `pull`, `pan-right`, `pan-left`, `tilt-up`, `tilt-down`.
Options de `render` : `--preset`, `--quality draft|high|ultra`, `--size 3840x2160`, `--fps`,
`--no-blur`, `--shutter 180`, `--start`, `--end`, `--no-audio`.

### Format du projet

Un exemple complet est fourni dans [`examples/push_in.project.json`](examples/push_in.project.json).
Les chemins relatifs sont résolus à partir du dossier du fichier projet.

| Champ | Rôle |
|-------|------|
| `camera.keyframes[].t` | instant en secondes, depuis le début de la plage exportée |
| `x`, `y` | centre de la caméra, 0 à 1 dans l'image source |
| `zoom` | 1 = cadrage de base, 2 = moitié de l'image visible (interpolé en logarithmique) |
| `rotation` | degrés, positif = sens horaire |
| `interp` | courbe du segment qui part de cette clé : `linear`, `ease_in`, `ease_out`, `ease_in_out`, `ease_in_cubic`, `ease_out_cubic`, `ease_in_out_cubic`, `ease_in_out_quint`, `smooth`, `hold`, ou `[x1, y1, x2, y2]` (Bézier façon CSS) |
| `output.fit` | `cover` (remplit le cadre) ou `contain` (image entière, bandes noires) |
| `motion_blur.shutter_angle` | 0 à 360° |
| `color.linear_light` | mélange du flou et du rééchantillonnage en lumière linéaire |

## Qualité : ce que fait le moteur

* **Flou de mouvement par suréchantillonnage temporel** : la caméra est évaluée à plusieurs instants
  pendant l'ouverture de l'obturateur. Le nombre d'échantillons s'adapte au déplacement réel de
  l'image en pixels (jusqu'à 96 en qualité ultra), donc pas de « marches d'escalier » sur les mouvements
  rapides et aucun coût inutile sur les mouvements lents.
* **Rééchantillonnage Lanczos** en virgule flottante pour les images nettes (bicubique quand le flou
  moyenne beaucoup d'échantillons : la différence disparaît dans le flou et c'est environ quatre fois
  plus rapide), avec **préfiltrage par aire** quand l'image est réduite (par exemple 4K vers 1080p),
  ce qui évite le scintillement des détails fins.
* **Lumière linéaire** pour mélanger les échantillons : un flou entre noir et blanc donne le bon gris
  perceptif au lieu d'un gris trop sombre.
* **Chaîne 10 bits** pour les préréglages H.265 10 bits, ProRes et FFV1 : le traitement reste en
  flottant jusqu'à l'encodeur, ce qui supprime le banding des dégradés. En 8 bits, un tramage
  stochastique évite le même défaut.
* **Couleurs maîtrisées** : matrice Rec.709 explicite à la lecture des sources non étiquetées en HD, et
  sortie étiquetée Rec.709.
* Sources 10 bits lues en 16 bits, images fixes (PNG, JPEG, TIFF, WebP, BMP) acceptées avec une durée
  au choix, rotation d'orientation des smartphones prise en compte, cadences NTSC exactes (29,97,
  59,94…).

### Préréglages d'export

| Clé | Format |
|-----|--------|
| `h264_high` | H.264 MP4, CRF 14, compatible partout |
| `h265_10bit` | H.265 10 bits MP4, CRF 16 |
| `prores_422hq` | Apple ProRes 422 HQ MOV, 10 bits (montage) |
| `prores_4444` | Apple ProRes 4444 MOV, 10 bits |
| `ffv1_lossless` | FFV1 MKV sans perte, 10 bits (archivage) |
| `h264_nvenc`, `hevc_nvenc` | encodage GPU NVIDIA (rapide) |

## Développement

```bash
pip install -r requirements.txt -r requirements-dev.txt
# FFmpeg doit être dans le PATH, ou dans vendor/ffmpeg (scripts/fetch_ffmpeg.ps1 sous Windows)
python -m pytest -q
python -m vmc gui            # interface
python -m vmc render projet.json
```

Structure :

```
src/vmc/curves.py       courbes d'accélération, spline monotone
src/vmc/camera.py       image clés, géométrie caméra → pixels
src/vmc/render.py       rééchantillonnage, flou de mouvement, conversion couleur
src/vmc/video_io.py     FFmpeg : détection, sondage, décodage et encodage par pipe
src/vmc/export.py       chaîne complète décodage → rendu → encodage
src/vmc/presets.py      préréglages FFmpeg
src/vmc/project.py      projet JSON
src/vmc/cli.py          ligne de commande
src/vmc/ai/             IA : modèles (téléchargement), moteur LivePortrait ONNX, chaîne vidéo
src/vmc/gui/            interface PySide6
tests/                  tests unitaires et de bout en bout (vrai FFmpeg)
vmc.spec                PyInstaller (deux exe dans un dossier)
scripts/                fetch_ffmpeg.ps1, build_windows.ps1
.github/workflows/      CI + build Windows + release
```

## Dépannage

* **« FFmpeg introuvable »** : le dossier `ffmpeg` doit rester à côté de l'exe (ou dans
  `_internal\ffmpeg`). Sinon, définissez `VMC_FFMPEG` avec le chemin de `ffmpeg.exe`.
* **Encodeur NVENC indisponible** : il faut une carte NVIDIA et des pilotes récents, sinon utilisez
  `h264_high`.
* **Windows SmartScreen / antivirus** : l'exe n'est pas signé. Choisissez « Informations complémentaires »
  puis « Exécuter quand même », ou signez-le avec votre certificat.
* **Rendu lent** : la qualité `ultra` et un obturateur à 360° multiplient les échantillons. Utilisez
  `draft` pour valider le mouvement, puis `high` ou `ultra` pour l'export final.
* Journal de l'interface : `%LOCALAPPDATA%\VideoMotionControl\log.txt`.

## Licences

Code sous licence MIT (voir [`LICENSE`](LICENSE)). **Modèles IA** : LivePortrait (MIT) ; les modèles de
détection de visage `retinaface_det_static` et `face_2dpose_106_static` viennent d'InsightFace et sont
réservés à un **usage non commercial**. Ne les utilisez pas dans un produit commercial sans licence. Le build Windows embarque FFmpeg (GPL), PySide6 (LGPL),
NumPy et OpenCV : voir [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
