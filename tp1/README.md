# TP1 — Mise en place de l'environnement de travail

## 1. Objectif

L'objectif de cette première partie du TP est de mettre en place l'environnement de travail nécessaire pour les séances de deep learning.

Cette mise en place comprend :

* l'utilisation de **SLURM** pour accéder aux ressources de calcul ;
* la réservation d'un GPU avec `srun` ;
* la soumission de jobs avec `sbatch` ;
* l'observation et l'annulation des jobs avec `squeue`, `scancel` et `sacct` ;
* la création d'un environnement Python isolé avec **Mamba** ;
* l'installation de **PyTorch avec le support CUDA** ;
* la vérification de l'accès au GPU depuis PyTorch ;
* la création d'un fichier `environment.yml` permettant de reproduire l'environnement.

---

# 2. Utilisation de SLURM

## 2.1 Connexion au cluster

L'accès au cluster se fait via SSH. La configuration SSH permet d'utiliser l'alias `tsp-client` pour se connecter au cluster.

La connexion est effectuée avec :

```bash
ssh tsp-client
```

La machine de connexion est un nœud de login. Elle ne doit pas être utilisée pour exécuter des programmes nécessitant beaucoup de ressources.

---

## 2.2 Réservation interactive d'un GPU

Pour obtenir des ressources de calcul, j'ai utilisé la commande :

```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --cpus-per-task=1 --mem=8G --pty bash
```

Cette commande demande :

* la partition `gpu` ;
* 1 GPU ;
* une durée maximale de 1 heure ;
* 1 CPU ;
* 8 Go de mémoire ;
* un shell interactif avec `--pty bash`.

Une fois le job attribué, le shell est exécuté sur un nœud de calcul possédant le GPU demandé.

---

## 2.3 Vérification du GPU avec `nvidia-smi`

Une fois connecté au nœud de calcul, la commande suivante permet de vérifier le GPU attribué :

```bash
nvidia-smi
```

Le GPU qui m'a été attribué est :

```text
NVIDIA L4
```

La commande `nvidia-smi` permet notamment de vérifier le modèle du GPU, son utilisation ainsi que sa mémoire disponible.

---

## 2.4 Observation et annulation d'un job

Pour afficher mes jobs SLURM en cours, j'ai utilisé :

```bash
squeue -u $USER
```

Cette commande permet notamment d'obtenir le `JobID` du job interactif.

### Job interactif

```text
JobID : TODO
```

Pour annuler un job, la commande utilisée est :

```bash
scancel MON_JOB_ID
```

Dans mon cas, la commande exacte utilisée était :

```bash
scancel TODO
```

> À compléter avec le véritable JobID utilisé pendant le TP.

---

# 3. Soumission d'un script avec `sbatch`

Pour tester le mode non interactif de SLURM, j'ai créé le fichier `hello.sh`.

Le script demande 1 GPU, 1 CPU et 8 Go de mémoire pendant une heure.

```bash
#!/bin/bash

#SBATCH --partition=gpu
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -J hello-slurm
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

mkdir -p logs

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"

nvidia-smi || echo "nvidia-smi indisponible"

echo "Bonjour depuis SLURM !"
```

Le script est soumis avec :

```bash
sbatch hello.sh
```

SLURM exécute alors le script indépendamment du terminal interactif.

Les sorties standard sont enregistrées dans le répertoire `logs/`.

Le nom du fichier de log généré est :

```text
TODO
```

> À compléter avec le nom exact obtenu dans `logs/`, par exemple `hello-slurm-XXXX.out`.

Le format du nom est défini par :

```text
logs/%x-%j.out
```

où `%x` correspond au nom du job et `%j` à son JobID.

---

# 4. Analyse des jobs avec `sacct`

Pour consulter l'historique d'un job terminé, la commande utilisée est :

```bash
sacct -j MON_JOB_ID --format=JobID,State,Elapsed,MaxRSS,ReqMem,ReqCPUS
```

Cette commande permet notamment de connaître :

* l'état du job ;
* sa durée d'exécution ;
* la mémoire réellement utilisée ;
* la mémoire demandée ;
* le nombre de CPUs demandés.

## Différence entre `ReqMem` et `MaxRSS`

`ReqMem` correspond à la quantité de mémoire **demandée lors de la réservation du job**.

`MaxRSS` correspond à la quantité maximale de mémoire **réellement utilisée par le job**.

Par exemple, si un job demande 8 Go mais n'utilise au maximum que 2 Go :

```text
ReqMem = 8G
MaxRSS = environ 2G
```

Ainsi, `ReqMem` décrit la ressource réservée tandis que `MaxRSS` permet d'observer la consommation réelle maximale de mémoire.

---

# 5. Transfert de fichiers

Les commandes `scp` et `rsync` permettent de transférer des fichiers entre ma machine et le cluster.

## 5.1 Machine → cluster

Pour envoyer un fichier vers le cluster :

```bash
scp ./mnist.zip tsp-client:~/data/
```

Le fichier local `mnist.zip` est alors copié dans :

```text
~/data/mnist.zip
```

sur le cluster.

---

## 5.2 Cluster → machine

Pour récupérer un fichier depuis le cluster :

```bash
scp tsp-client:~/results/output.log ./output.log
```

Le fichier distant :

```text
~/results/output.log
```

est copié dans le répertoire courant de ma machine sous le nom :

```text
output.log
```

---

## 5.3 Synchronisation avec `rsync`

Pour synchroniser un dossier local avec un dossier distant :

```bash
rsync -avhP ./results/ tsp-client:~/backup-results/
```

`rsync` permet notamment d'éviter de retransférer inutilement les fichiers qui n'ont pas changé.

---

# 6. Création de l'environnement Python

## 6.1 Création de l'environnement

Un environnement dédié au cours a été créé avec Mamba :

```bash
mamba create -n deeplearning python=3.10
```

Puis il a été activé avec :

```bash
mamba activate deeplearning
```

Cet environnement permet d'isoler les dépendances du TP du reste du système.

---

## 6.2 Vérification de Python

Pour vérifier la version de Python utilisée ainsi que le chemin du binaire :

```bash
python --version
which python
```

Résultat :

```text
Python 3.10.21
```

Chemin du binaire :

```text
/mnt/hdd/homes/imrad/miniforge3/envs/deeplearning/bin/python
```

L'environnement utilise donc bien Python 3.10 et le binaire provient de l'environnement `deeplearning`.

---

# 7. Installation de PyTorch et CUDA

PyTorch a été installé avec le support CUDA à l'aide de Mamba.

Les composants principaux de l'environnement sont notamment :

```text
PyTorch        2.5.1
torchvision    0.20.1
pytorch-cuda   12.1
Python         3.10.21
```

L'environnement utilise CUDA 12.1 côté runtime PyTorch.

Le GPU du nœud de calcul est une NVIDIA L4.

---

# 8. Vérification de PyTorch et CUDA

Le fichier `check_gpu.py` contient :

```python
import torch

print("PyTorch version:", torch.__version__)

gpu_available = torch.cuda.is_available()

print("CUDA available:", gpu_available)

if gpu_available:
    print("Device count:", torch.cuda.device_count())
    print("Device 0 name:", torch.cuda.get_device_name(0))
else:
    print("Attention, aucun GPU détecté !")
```

Il est exécuté avec :

```bash
python check_gpu.py
```

### Résultat

```text
TODO
```

> Coller ici la sortie réelle obtenue avec `python check_gpu.py`.

Le résultat attendu lorsque l'environnement est correctement configuré est notamment :

```text
PyTorch version: 2.5.1
CUDA available: True
Device count: 1
Device 0 name: NVIDIA L4
```

---

## Si `CUDA available` vaut `False`

Deux causes possibles sont par exemple :

1. le programme est exécuté sur la machine de connexion au lieu d'un nœud de calcul disposant d'un GPU ;
2. PyTorch a été installé avec une version ne disposant pas du support CUDA ou l'environnement CUDA/PyTorch est mal configuré.

Il faut donc notamment vérifier que le programme est exécuté dans une allocation SLURM possédant un GPU et que `torch.cuda.is_available()` retourne `True`.

---

# 9. Reproductibilité de l'environnement

Pour enregistrer les dépendances principales de l'environnement, la commande suivante a été utilisée :

```bash
mamba env export --from-history -n deeplearning > environment.yml
```

Le fichier obtenu est :

```text
environment.yml
```

Il est placé dans le répertoire `TP1` et ajouté au dépôt Git.

Ce fichier permet de conserver une description reproductible des dépendances utilisées pour le TP.

---

# 10. Vérification de TensorBoard

TensorBoard a été installé avec :

```bash
mamba install tensorboard -c conda-forge
```

La commande permettant normalement d'afficher sa version est :

```bash
tensorboard --version
```

Dans mon environnement, cette commande a rencontré l'erreur suivante :

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

Bien que TensorBoard soit installé dans l'environnement, son lancement avec cette commande rencontre donc actuellement un problème de dépendance lié à `pkg_resources`.

Version installée de TensorBoard :

```text
2.20.0
```

Ce problème sera corrigé avant l'utilisation de TensorBoard dans la suite du TP.

---

# 11. Fichiers créés à ce stade

À ce stade du TP, le répertoire `TP1` contient notamment :

```text
TP1/
├── rapport.md
├── hello.sh
├── check_gpu.py
├── environment.yml
└── logs/
    └── TODO
```

Les fichiers sont ajoutés progressivement au dépôt Git conformément aux modalités du TP.

---

# 12. Bilan

Cette première partie a permis de mettre en place l'environnement nécessaire pour les prochains exercices.

J'ai notamment appris à :

* réserver des ressources avec `srun` ;
* obtenir un GPU NVIDIA L4 ;
* observer mes jobs avec `squeue` ;
* annuler un job avec `scancel` ;
* soumettre un script avec `sbatch` ;
* consulter l'historique avec `sacct` ;
* transférer des fichiers avec `scp` ;
* synchroniser des dossiers avec `rsync` ;
* créer un environnement Python avec Mamba ;
* installer PyTorch et CUDA ;
* vérifier la disponibilité du GPU depuis PyTorch ;
* exporter l'environnement dans `environment.yml`.

La suite du TP portera sur les notions théoriques des réseaux de neurones puis sur l'entraînement d'un MLP sur CIFAR-10.
