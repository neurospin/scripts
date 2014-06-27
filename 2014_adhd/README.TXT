masque: /neurospin/adhd200/python_analysis/data/mask_t0.1_sum0.8_closing.nii
je ne sais plus si j'ai copié le template quelque part, il est au moins dans le dossier du premier sujet KKI c'est le Template_6.nii

Les gros tableaux numpy sont stockés dans des fichiers HDF5 dans le répertoire /neurospin/adhd200/python_analysis/data
pour chaque feature tu as un fichier train et un fichier test, le préfixe indique de quelle feature il s'agit, exemples les plus intéressants:
jd => jacobian determinant
w => GM warped
mw => GM warped modulated
lt => log tensor
ltw => log tensor combined with warped GM
s'il y a un s devant c'est la version lissée

les fichiers s'ouvrent avec tables:
  import tables
  h5_file = tables.openFile(filepath)

pour récupérer le diagnostic:
  Y = h5_file.root.DX.read()
donne un tableau numpy de taille (n_subjects) avec 0 pour controle, 1 pour ADHD, et (attention!) -1 pour diagnostic inconnu (quelques un dans le groupe test)
avec DX3 au lieu de DX tu as le diagnostic avec les sous-types (1,2,3 correspondent aux sous-types, je ne sais plus lequel est lequel mais ça se retrouve)

pour récupérer les données:
  X = h5_file.root.X
tu te retrouves avec un objet compatible avec numpy (sauf slicing compliqué) qui lit les données quand il y en a besoin -- pour charger le tableau complètement en mémoire:
  X = h5_file.root.X.read()
et tu te retrouves avec un tableau numpy avec les données masquées
pour les données univariées: taille (n_subjects, n_voxels)
pour les données multivariées: taille (n_subjects, n_voxels, n_dimensions) où n_dimensions est 6 pour les tenseurs, par exemple
