import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Fixer la graine pour la reproductibilité

np.random.seed(0)

# Générer des données
X = np.random.normal(0, 2, 100)
Y = 0.5 * X + np.random.normal(0, 1, 100)

# Calcul de la covariance
covariance = np.cov(X, Y)[0, 1]

# Calcul des moyennes
mean_x, mean_y = np.mean(X), np.mean(Y)

# Création du graphique
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, alpha=0.6, label='Données')

# Ajouter les lignes des moyennes
plt.axvline(x=mean_x, color='red', linestyle='dashed', label='Moyenne de X')
plt.axhline(y=mean_y, color='blue', linestyle='dashed', label='Moyenne de Y')

# Marquer le centre
plt.scatter(mean_x, mean_y, color='green', s=100, label='Centre (E[X], E[Y])')

# Annotation pour la covariance
plt.annotate(f'Covariance: {covariance:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
             ha='left', va='top', fontsize=12, color='black')

# Quadrants
plt.fill_betweenx([mean_y, 10], mean_x, 10, color='gray', alpha=0.3)
plt.fill_betweenx([-10, mean_y], -10, mean_x, color='gray', alpha=0.3)

# Configurations du graphique
plt.title('Visualisation de la Covariance entre X et Y', fontsize=16)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

# Affichage du graphique
plt.show()

"""## **Visualisation de la Matrice de Covariance d'une Gausienne Multivariée**"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_covariance_matrix(mean, cov):
    fig, ax = plt.subplots()
    # Générer des données aléatoires
    data = np.random.multivariate_normal(mean, cov, size=500)
    ax.scatter(data[:,0], data[:,1], alpha=0.5)

    # Calculer les valeurs propres et les vecteurs propres pour la matrice de covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Trier les valeurs propres et les vecteurs propres
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Tracer l'ellipse
    for j in range(1, 4):  # Augmenter j pour couvrir plus d'écarts-types
        ell = Ellipse(xy=mean,
                      width=2 * j * np.sqrt(eigenvalues[0]), height=2 * j * np.sqrt(eigenvalues[1]),
                      angle=np.rad2deg(np.arctan2(*eigenvectors[:, 0][::-1])),
                      edgecolor='red', fc='None', lw=2)
        ax.add_patch(ell)

    # Ajouter une légende une seule fois
    ax.plot([], [], 'r-', label=f'λ1: {eigenvalues[0]:.2f}, λ2: {eigenvalues[1]:.2f}')

    # Tracer les vecteurs propres et annoter avec les valeurs propres
    for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        vector_multiplier = 2 * np.sqrt(eigval)
        vector_end = mean + eigvec * vector_multiplier
        ax.arrow(mean[0], mean[1], eigvec[0] * vector_multiplier, eigvec[1] * vector_multiplier,
                 head_width=0.2, head_length=0.3, fc='blue', ec='blue')
        ax.text(vector_end[0], vector_end[1], f'$\lambda_{i+1} = {eigval:.2f}$', color='blue', ha='center')

    ax.set_title("Visualisation de la Matrice de Covariance")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

# Moyenne et matrice de covariance
mean = [2, 3]
covariance_matrix = [[2, 0.5], [0.5, 1]]
plot_covariance_matrix(mean, covariance_matrix)

"""## **Visualisation d'une Gausienne Multivariée (Gausienne Bidimensionnelle)**"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d  # Importation correcte pour art3d

# Définir la moyenne et la matrice de covariance de la gaussienne multivariée
mean = [0, 0]
covariance = [[1, 0.5], [0.5, 1]]

# Générer des échantillons aléatoires de la gaussienne multivariée
data = np.random.multivariate_normal(mean, covariance, size=1000)

# Créer une grille pour le tracé 3D
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
rv = multivariate_normal(mean, covariance)
Z = rv.pdf(pos)

# Valeurs propres et vecteurs propres pour l'ellipse
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

# Préparation du graphique
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracé des points de données
ax.scatter(data[:, 0], data[:, 1], zs=0, zdir='z', s=20, c='k', depthshade=True)

# Tracé de la surface pour la gaussienne
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

# Ellipse représentant la covariance
ellipse = Ellipse(xy=mean, width=2*np.sqrt(eigenvalues[0]), height=2*np.sqrt(eigenvalues[1]), angle=angle,
                  edgecolor='red', facecolor='none', linestyle='--')
ax.add_patch(ellipse)
art3d.pathpatch_2d_to_3d(ellipse, z=0, zdir="z")

# Ajout des histogrammes
histx, bin_edges = np.histogram(data[:, 0], bins=20, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax.bar(bin_centers, histx, zs=-4, zdir='y', align='center', color='r', alpha=0.7)

histy, bin_edges = np.histogram(data[:, 1], bins=20, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax.bar(bin_centers, histy, zs=4, zdir='x', align='center', color='b', alpha=0.7)

# Étiquettes et titre
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Densité de probabilité')
ax.set_title('Visualisation 3D d\'une distribution gaussienne multivariée avec ellipse de confiance')

plt.show()

"""# **Loi des Grands Nombres et Théorème Central Limite**

## **Visualisation de la Loi Forte des Grands Nombres**
"""

import numpy as np
import matplotlib.pyplot as plt

# Fixer la graine pour la reproductibilité
np.random.seed(42)

# Paramètres
n_trials = 1500  # Nombre de lancers
coin_flips = np.random.choice([0, 1], size=n_trials)  # 0 = pile, 1 = face

# Calculer la proportion cumulative de 'face'
cumulative_heads = np.cumsum(coin_flips) / (np.arange(1, n_trials + 1))

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(cumulative_heads, label="Proportion Cumulative de 'Face'", color='blue')
plt.axhline(y=0.5, color='red', linestyle='--', label="Valeur Théorique ($\\frac{1}{2}$)")
plt.title("Convergence de la Proportion de 'Face' vers $\\frac{1}{2}$", fontsize=14)
plt.xlabel("Nombre de Lancers", fontsize=12)
plt.ylabel("Proportion Cumulative", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

"""## **Visualisation du Théorème Central Limite**"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Fixer la graine pour la reproductibilité
np.random.seed(8302)

# Paramètres
n_samples = 1000  # Nombre d'échantillons
sample_sizes = [5, 30, 100]  # Différentes tailles d'échantillons
true_mean, true_variance = 3, 2  # Moyenne et variance de la distribution d'origine

# Génération des échantillons
original_data = np.random.exponential(scale=true_mean, size=(max(sample_sizes), n_samples))

# Visualisation
plt.figure(figsize=(12, 8))
for i, size in enumerate(sample_sizes):
    sample_means = np.mean(original_data[:size, :], axis=0)  # Moyenne des échantillons
    plt.subplot(1, len(sample_sizes), i + 1)
    plt.hist(sample_means, bins=30, density=True, alpha=0.7, label=f'Taille = {size}')
    x = np.linspace(min(sample_means), max(sample_means), 100)
    plt.plot(x, norm.pdf(x, loc=true_mean, scale=np.sqrt(true_variance / size)),
             label="Distribution Normale", color='red')
    plt.title(f"Moyenne avec n={size}")
    plt.xlabel("Valeurs Moyennes")
    plt.ylabel("Densité")
    plt.legend()

plt.tight_layout()
plt.show()