import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from math import radians, sin, cos, atan2, sqrt
import folium
from streamlit_folium import st_folium

# ==========================
# Funciones de apoyo
# ==========================

def distancia_haversine(lat1, lon1, lat2, lon2):
    """Distancia en km entre dos puntos usando Haversine."""
    radio_tierra = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * radio_tierra * atan2(sqrt(a), sqrt(1-a))

def construir_grafo_knn(df, k=3):
    """Construye un grafo k-NN simple a partir de lat/long."""
    G = nx.Graph()
    coords = {}

    for _, fila in df.iterrows():
        ruc = str(fila["RUC"])
        lat = float(fila["LATITUD"])
        lon = float(fila["LONGITUD"])
        G.add_node(ruc, nombre=fila["RAZON_SOCIAL"], lat=lat, lon=lon)
        coords[ruc] = (lat, lon)

    for nodo in coords:
        distancias = []
        for otro in coords:
            if otro == nodo:
                continue
            d = distancia_haversine(coords[nodo][0], coords[nodo][1],
                                    coords[otro][0], coords[otro][1])
            distancias.append((otro, d))
        distancias.sort(key=lambda x: x[1])
        for vecino, d in distancias[:k]:
            G.add_edge(nodo, vecino, weight=d)
    return G

def construir_grafo_mst(df):
    """Construye un MST sobre todas las distancias (puede ser pesado si hay demasiados nodos)."""
    G_completo = nx.Graph()
    coords = {}

    for _, fila in df.iterrows():
        ruc = str(fila["RUC"])
        lat = float(fila["LATITUD"])
        lon = float(fila["LONGITUD"])
        G_completo.add_node(ruc, nombre=fila["RAZON_SOCIAL"], lat=lat, lon=lon)
        coords[ruc] = (lat, lon)

    nodos = list(coords.keys())
    # grafo completo
    for i in range(len(nodos)):
        for j in range(i+1, len(nodos)):
            n1, n2 = nodos[i], nodos[j]
            d = distancia_haversine(coords[n1][0], coords[n1][1],
                                    coords[n2][0], coords[n2][1])
            G_completo.add_edge(n1, n2, weight=d)

    MST = nx.minimum_spanning_tree(G_completo, weight="weight", algorithm="kruskal")
    return MST

def dibujar_grafo_spring(G):
    """Dibujo abstracto del grafo (spring layout) con NetworkX."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, k=0.15, iterations=30)
    nx.draw(G, pos, node_size=20, node_color="skyblue",
            edge_color="gray", with_labels=False, ax=ax)
    ax.set_title(f"Grafo – {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    ax.axis("off")
    return fig

def dibujar_mapa_folium(G):
    """Mapa Folium con nodos y aristas."""
    if G.number_of_nodes() == 0:
        return None

    lats = [G.nodes[n]["lat"] for n in G.nodes]
    lons = [G.nodes[n]["lon"] for n in G.nodes]
    centro = [np.mean(lats), np.mean(lons)]

    m = folium.Map(location=centro, zoom_start=11, control_scale=True)

    # Aristas
    for u, v, data in G.edges(data=True):
        lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
        lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
        folium.PolyLine([(lat1, lon1), (lat2, lon2)], weight=2, opacity=0.7).add_to(m)

    # Nodos
    for n, attr in G.nodes(data=True):
        popup = f"<b>{attr.get('nombre','')}</b><br>RUC: {n}"
        folium.CircleMarker(
            location=[attr["lat"], attr["lon"]],
            radius=4, fill=True, fill_opacity=0.9, color="black", weight=0.5,
            fill_color="#8FEAF3"
        ).add_to(m).add_child(folium.Popup(popup, max_width=250))

    return m

def calcular_ruta_dijkstra(G, origen, destino):
    try:
        camino = nx.shortest_path(G, source=origen, target=destino, weight="weight")
        longitud = nx.shortest_path_length(G, source=origen, target=destino, weight="weight")
        return camino, longitud
    except nx.NetworkXNoPath:
        return None, None

# ==========================
# Configuración de Streamlit
# ==========================

st.set_page_config(page_title="Optimizador Logístico Courier", layout="wide")
st.title("📦 Optimizador Logístico Courier con Grafos")

st.sidebar.header("⚙️ Configuración del aplicativo")

tipo_grafo = st.sidebar.selectbox(
    "Tipo de grafo",
    ["k-NN", "MST"],
    key="sb_tipo_grafo"
)

k_vecinos = st.sidebar.slider(
    "k vecinos (solo k-NN)",
    min_value=1,
    max_value=6,
    value=3,
    step=1,
    key="sb_k_vecinos"
)

submuestro = st.sidebar.checkbox(
    "Usar submuestreo visual",
    value=True,
    key="sb_submuestreo"
)

n_max = st.sidebar.slider(
    "Máx. nodos a visualizar",
    min_value=100,
    max_value=1500,
    value=400,
    step=100,
    key="sb_n_max"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Análisis disponibles")
activar_ruta = st.sidebar.checkbox("Ruta óptima (Dijkstra)", key="sb_ruta")
activar_hubs = st.sidebar.checkbox("Hubs (betweenness)", key="sb_hubs")
activar_falla = st.sidebar.checkbox("Simulación de falla", key="sb_falla")
activar_drones = st.sidebar.checkbox("Escenario con drones", key="sb_drones")


# ==========================
# Lógica principal
# ==========================

# ==========================
# Lógica principal
# ==========================

st.sidebar.markdown("### 📂 Base de datos fija")
st.sidebar.markdown("Usando archivo: **DataBase.xlsx**")

# Leer datos directamente del archivo local
DATA_PATH = "DataBase.xlsx"   # nombre de tu archivo en la misma carpeta

try:
    df = pd.read_excel(DATA_PATH)
except FileNotFoundError:
    st.error(f"No se encontró el archivo {DATA_PATH}. Asegúrate de que esté en la misma carpeta que app.py.")
    st.stop()

# Asumimos nombres estándar; ajusta si difieren
df = df[["RUC", "RAZON_SOCIAL", "LATITUD", "LONGITUD"]].copy()
df = df.dropna(subset=["LATITUD","LONGITUD"])
df = df.drop_duplicates(subset=["RUC"])
df["LATITUD"] = df["LATITUD"].astype(float)
df["LONGITUD"] = df["LONGITUD"].astype(float)

st.success(f"Datos cargados correctamente desde {DATA_PATH}. Registros válidos: {len(df)}")


# Asumimos nombres estándar; ajusta si difieren
df = df[["RUC", "RAZON_SOCIAL", "LATITUD", "LONGITUD"]].copy()
df = df.dropna(subset=["LATITUD","LONGITUD"])
df = df.drop_duplicates(subset=["RUC"])
df["LATITUD"] = df["LATITUD"].astype(float)
df["LONGITUD"] = df["LONGITUD"].astype(float)

st.success(f"Datos cargados correctamente. Registros válidos: {len(df)}")

# Submuestreo visual
if submuestro and len(df) > n_max:
    df_vis = df.sample(n_max, random_state=42).reset_index(drop=True)
else:
    df_vis = df.copy()

# Construir grafo según selección
if tipo_grafo == "k-NN":
    G = construir_grafo_knn(df_vis, k=k_vecinos)
else:
    st.warning("MST puede demorar si hay muchos nodos; úsalo con <= 200 nodos.")
    G = construir_grafo_mst(df_vis)

# ==========================
# Tabs de interfaz
# ==========================

tab_dataset, tab_grafo, tab_mapa, tab_rutas, tab_hubs, tab_fallas, tab_drones = st.tabs(
    ["📄 Dataset", "🕸 Grafo", "🗺 Mapa", "🧭 Rutas", "⭐ Hubs", "⚠️ Fallas", "🚁 Drones"]
)

# -------- Tab Dataset --------
with tab_dataset:
    st.subheader("Vista del dataset")
    st.dataframe(df.head(20))
    st.write(f"Total de nodos (RUC únicos): {df['RUC'].nunique()}")

# -------- Tab Grafo --------
with tab_grafo:
    st.subheader("Grafo (vista abstracta)")
    fig = dibujar_grafo_spring(G)
    st.pyplot(fig)

# -------- Tab Mapa --------
with tab_mapa:
    st.subheader("Grafo georreferenciado")
    mapa = dibujar_mapa_folium(G)
    if mapa:
        st_folium(mapa, width=900, height=600)
    else:
        st.warning("No se pudo construir el mapa.")

# -------- Tab Rutas --------
with tab_rutas:
    st.subheader("Cálculo de ruta óptima (Dijkstra)")
    if not activar_ruta:
        st.info("Activa 'Ruta óptima (Dijkstra)' en la barra lateral.")
    else:
        nodos = list(G.nodes)
        if len(nodos) < 2:
            st.warning("No hay suficientes nodos para calcular rutas.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                origen = st.selectbox("Nodo origen (RUC)", nodos)
            with col2:
                destino = st.selectbox("Nodo destino (RUC)", [n for n in nodos if n != origen])

            if st.button("Calcular ruta"):
                camino, dist_km = calcular_ruta_dijkstra(G, origen, destino)
                if camino:
                    st.success(f"Camino encontrado ({len(camino)} nodos), distancia aprox: {dist_km:.2f} km")
                    st.write("Ruta:", " → ".join(camino))
                else:
                    st.error("No existe ruta entre esos nodos en el grafo.")

# -------- Tab Hubs --------
with tab_hubs:
    st.subheader("Análisis de hubs logísticos")
    if not activar_hubs:
        st.info("Activa 'Hubs (betweenness)' en la barra lateral.")
    else:
        if G.number_of_nodes() == 0:
            st.warning("No hay nodos.")
        else:
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
            df_bc = pd.DataFrame([
                {"RUC": n, "Razon_Social": G.nodes[n].get("nombre",""), "Betweenness": v}
                for n, v in bc.items()
            ])
            df_bc = df_bc.sort_values("Betweenness", ascending=False).head(10)
            st.write("Top 10 nodos por betweenness:")
            st.dataframe(df_bc)

# -------- Tab Fallas --------
with tab_fallas:
    st.subheader("Simulación de falla de nodo")
    if not activar_falla:
        st.info("Activa 'Simulación de falla' en la barra lateral.")
    else:
        nodos = list(G.nodes)
        if len(nodos) == 0:
            st.warning("No hay nodos para simular.")
        else:
            victima = st.selectbox("Nodo a desactivar (RUC)", nodos)
            if st.button("Simular falla"):
                G_fail = G.copy()
                G_fail.remove_node(victima)
                comps = list(nx.connected_components(G_fail))
                st.write(f"Componentes conectados tras la falla: {len(comps)}")
                st.write(f"Ejemplo de componente aislado (si existe): {list(comps[0])[:10]}")
                # Aquí podrías añadir un mapa post-falla si quieres.

# -------- Tab Drones --------
with tab_drones:
    st.subheader("Escenario de uso de drones")
    if not activar_drones:
        st.info("Activa 'Escenario con drones' en la barra lateral.")
    else:
        st.write("Aquí puedes estimar energía, autonomía y comparar rutas con vs sin dron.")
        # TODO: aquí enchufas tus fórmulas de Wh/km, autonomía, etc.
        st.markdown("- Ejemplo: consumo base 15 Wh/km + 1.2 Wh/km·kg por carga.")
        st.markdown("- Autonomía estimada: distancia máxima antes de recarga.")

st.sidebar.header("⚙️ Configuración del aplicativo")

tipo_grafo = st.sidebar.selectbox("Tipo de grafo", ["k-NN", "MST"])
k_vecinos = st.sidebar.slider("k vecinos (solo k-NN)", 1, 6, 3)




