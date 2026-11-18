import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from numba import njit

#Funções Otimizadas com Numba
@njit
def calcular_scores(X, eigenvectors_components):
    return X @ eigenvectors_components

#Carregar os dados e Calcular Anomalias
Arq = "/content/sst.mnmean.nc"
ds = xr.open_dataset(Arq)
sst = ds['sst']
climatology = sst.groupby('time.month').mean('time')
anomalies = sst.groupby('time.month') - climatology

#Preparação dos dados para PCA
stacked_data = anomalies.stack(space=('lat', 'lon'))
stacked_data_filled = stacked_data.fillna(0)
X = stacked_data_filled.values
n_samples, n_features = X.shape

#Conversão crítica para Numba
X = X.astype(np.float64)

#Implementação da PCA com NumPy e Numba
covariance_matrix = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Ordenação
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]
n_components = 5

# Cálculo dos Scores
pca_scores = calcular_scores(X, eigenvectors_sorted[:, :n_components])
pca_components_flat = eigenvectors_sorted[:, :n_components].T

#Desempilhamento temporário para obter o EOF1 (Componente 1)
space_multiindex = stacked_data_filled.indexes['space']

#Constrói o DataArray do Componente 1 (EOF1)
component1_flat_xr = xr.DataArray(
    pca_components_flat[0, :],
    coords={'space': space_multiindex},
    dims=['space']
)
eof1_map_temp = component1_flat_xr.unstack('space')

try:
    target_value = eof1_map_temp.sel(lat=0, lon=270, method='nearest').item()
except KeyError:
    #Se o ponto exato não existir, pegamos uma área e tiramos a média
    target_value = eof1_map_temp.sel(lat=slice(5, -5), lon=slice(240, 290)).mean().item()

#Se o valor for negativo (o que implica um sinal invertido), multiplicamos por -1
if target_value < 0:
    print("\nSinal do EOF1 invertido detectado no Pacífico Leste. Corrigindo...")
    pca_scores[:, 0] = pca_scores[:, 0] * -1
    pca_components_flat[0, :] = pca_components_flat[0, :] * -1
else:
    print("\nSinal do EOF1 OK (Positivo no Pacífico Leste). Nenhuma correção necessária.")

#Re-cria o DataArray dos 5 componentes corrigidos
components_xr = xr.DataArray(
    pca_components_flat,
    coords={
        'mode': np.arange(1, n_components + 1),
        'space': space_multiindex
    },
    dims=['mode', 'space']
)

componentes_mapa = components_xr.unstack('space')

#Plotar o "Score" (Série Temporal do PC1)
pc1_timeseries = xr.DataArray(pca_scores[:, 0], coords={'time': anomalies['time']}, dims='time')

plt.figure(figsize=(12, 4))
pc1_timeseries.plot()
plt.title('Componente PCA 1 (Índice ENSO)')
plt.xlabel('Ano')
plt.ylabel('Amplitude (adimensional)')
plt.grid(True)
plt.show()

#Plotar o "Componente" (Padrão Espacial do EOF1)
eof1_map = componentes_mapa.sel(mode=1)

plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

eof1_map.plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',
    cbar_kwargs={'label': 'Padrão EOF (adimensional)'}
)
ax.coastlines()
ax.gridlines(draw_labels=True)
plt.title('Padrão Espacial 1 (Padrão ENSO/El Niño)')
plt.show()
