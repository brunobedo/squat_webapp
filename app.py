"""Análise Markerless do Agachamento — webapp Streamlit.

Processa um vídeo do agachamento (vista sagital) com MediaPipe Pose,
calcula os ângulos de quadril e joelho e apresenta:
  1. envio do vídeo (upload ou exemplo);
  2. vídeo processado com play e gráfico de ângulos sincronizados lado a lado;
  3. tabela de métricas do agachamento (picos, amplitude, velocidade angular);
  4. downloads do CSV de ângulos e do vídeo processado.
"""

import base64
import json
import os
import tempfile
from string import Template

import cv2
import imageio.v2 as imageio
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.maintools import calculate_angle

# ----------------------------------------------------------------------------
# Configuração da página e estilo
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Análise Markerless dos Membros Inferiores (ex: agachamento)",
    page_icon="◾",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* Trava o tema claro mesmo se o navegador preferir modo escuro */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #ffffff !important;
    }
    .stApp p, .stApp label, .stApp span, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #000000 !important;
    }
    [data-testid="stAlertContainer"] {
        background: #ffffff !important;
        border: 1px solid #000000;
        border-radius: 0;
    }
    [data-testid="stAlertContainer"] p {color: #000000 !important;}
    [data-testid="stExpander"] details {
        background: #ffffff;
        border: 1px solid #000000;
        border-radius: 0;
    }
    [data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: #000000 !important;
        border-radius: 0;
        color: #000000 !important;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 3rem;
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #000;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        font-weight: 300;
        color: #000;
        margin-bottom: 0.5rem;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #000;
        border-bottom: 1px solid #000;
        padding-bottom: 0.4rem;
        margin: 2.2rem 0 1rem 0;
    }
    hr {border-color: #e5e5e5;}

    /* Botões de ação e de download em cinza */
    .stButton > button, .stDownloadButton > button {
        background: #808080 !important;
        color: #fff !important;
        border: 1px solid #808080 !important;
        border-radius: 0;
        padding: 0.55rem 1.6rem;
        font-weight: 500;
        letter-spacing: 0.05em;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #5a5a5a !important;
        border: 1px solid #5a5a5a !important;
        color: #fff !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #fff;
        border: 1px dashed #000;
        border-radius: 0;
    }

    [data-testid="stCaptionContainer"], .stCaption, small {
        color: #000 !important;
    }

    .citacao-box {
        border: 1px solid #000;
        background: #fff;
        padding: 1.4rem 1.6rem;
        margin-top: 2.5rem;
    }
    .citacao-titulo {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #000;
        margin-bottom: 0.7rem;
    }
    .citacao-texto {
        font-size: 0.85rem;
        line-height: 1.65;
        color: #000;
        margin-bottom: 1rem;
    }
    .citacao-links {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
    }
    .citacao-link {
        display: inline-block;
        background: #000;
        color: #fff !important;
        border: 1px solid #000;
        padding: 0.45rem 1.1rem;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-decoration: none !important;
    }
    .citacao-link:hover {
        background: #fff;
        color: #000 !important;
    }
    .citacao-link.secundario {
        background: #fff;
        color: #000 !important;
    }
    .citacao-link.secundario:hover {
        background: #000;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Processamento markerless (adaptado de run.py / src/maintools.py)
# ----------------------------------------------------------------------------


def _desenhar_anotacoes(image, results, landmarks_px, angle_knee, angle_hip):
    """Desenha esqueleto e valores de ângulo no quadro (estética monocromática)."""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=6, circle_radius=2),
    )

    for texto, ponto in (
        (f"Joelho: {angle_knee:.1f}", landmarks_px["knee"]),
        (f"Quadril: {angle_hip:.1f}", landmarks_px["hip"]),
    ):
        cv2.putText(image, texto, ponto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, texto, ponto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def processar_video(video_path, side, min_confidence, progress_callback=None):
    """Roda o MediaPipe Pose no vídeo e retorna (df, video_path, fps).

    - df: DataFrame com Frame, Knee_Angle e Hip_Angle (NaN quando não detectado);
    - video_path: caminho do vídeo processado em H.264 (compatível com navegador).
    """
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo enviado.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # yuv420p exige dimensões pares
    out_w, out_h = frame_width - frame_width % 2, frame_height - frame_height % 2

    output_path = os.path.join(tempfile.gettempdir(), "agachamento_processado.mp4")
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    knee_angles, hip_angles, frame_ids = [], [], []

    if side == "r":
        lm_ids = {
            "hip": mp_pose.PoseLandmark.RIGHT_HIP,
            "knee": mp_pose.PoseLandmark.RIGHT_KNEE,
            "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
            "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        }
    else:
        lm_ids = {
            "hip": mp_pose.PoseLandmark.LEFT_HIP,
            "knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
            "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        }

    with mp_pose.Pose(
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    ) as pose:
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            angle_knee, angle_hip = np.nan, np.nan
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                pts = {k: [lm[v.value].x, lm[v.value].y] for k, v in lm_ids.items()}

                angle_knee = 180 - calculate_angle(pts["hip"], pts["knee"], pts["ankle"])
                angle_hip = 180 - calculate_angle(pts["shoulder"], pts["hip"], pts["knee"])

                px = {
                    k: tuple(np.multiply(v, [frame_width, frame_height]).astype(int))
                    for k, v in pts.items()
                }
                _desenhar_anotacoes(image, results, px, angle_knee, angle_hip)

            knee_angles.append(angle_knee)
            hip_angles.append(angle_hip)
            frame_ids.append(frame_id)

            writer.append_data(cv2.cvtColor(image[:out_h, :out_w], cv2.COLOR_BGR2RGB))

            frame_id += 1
            if progress_callback:
                progress_callback(min(frame_id / total_frames, 1.0))

    cap.release()
    writer.close()

    df = pd.DataFrame({"Frame": frame_ids, "Knee_Angle": knee_angles, "Hip_Angle": hip_angles})
    return df, output_path, fps


# ----------------------------------------------------------------------------
# Player sincronizado (vídeo + gráfico lado a lado)
# ----------------------------------------------------------------------------

_TEMPLATE_PLAYER = Template(
    """
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { margin: 0; font-family: 'Inter', 'Helvetica Neue', sans-serif; }
</style>
<div id="wrap" style="display:flex; gap:16px; align-items:stretch;">
  <div style="flex:1; min-width:0; border:1px solid #e5e5e5; background:#000;">
    <video id="vid" controls muted playsinline
           style="width:100%; height:${altura}px; object-fit:contain; display:block; background:#000;">
      <source src="data:video/mp4;base64,${video_b64}" type="video/mp4">
    </video>
  </div>
  <div style="flex:1; min-width:0; border:1px solid #e5e5e5;">
    <div id="chart" style="width:100%; height:${altura}px;"></div>
  </div>
</div>
<div id="leitura" style="margin-top:10px; font-size:13.5px; color:#000; letter-spacing:0.02em;"></div>
<script>
  const knee = ${knee};
  const hip = ${hip};
  const fps = ${fps};
  const n = knee.length;
  const frames = [...Array(n).keys()];

  const data = [
    {x: frames, y: knee, name: 'Joelho', mode: 'lines',
     line: {color: '#000000', width: 2.5}},
    {x: frames, y: hip, name: 'Quadril', mode: 'lines',
     line: {color: '#888888', width: 2.5, dash: 'dash'}},
    {x: [0], y: [knee[0]], mode: 'markers', showlegend: false, hoverinfo: 'skip',
     marker: {color: '#000000', size: 11, line: {color: '#ffffff', width: 2}}},
    {x: [0], y: [hip[0]], mode: 'markers', showlegend: false, hoverinfo: 'skip',
     marker: {color: '#888888', size: 11, line: {color: '#ffffff', width: 2}}}
  ];

  const layout = {
    font: {family: 'Inter, Helvetica, sans-serif', color: '#000000', size: 12},
    xaxis: {title: 'Quadro', gridcolor: '#eeeeee', range: [0, n - 1], zeroline: false},
    yaxis: {title: 'Ângulo (graus)', gridcolor: '#eeeeee', range: [0, 185], zeroline: false},
    legend: {orientation: 'h', y: 1.06, x: 1, xanchor: 'right'},
    margin: {l: 55, r: 15, t: 10, b: 45},
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    shapes: [{type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
              line: {color: '#000000', width: 1.5, dash: 'dot'}}]
  };

  Plotly.newPlot('chart', data, layout, {displayModeBar: false, responsive: true});

  const vid = document.getElementById('vid');
  const chart = document.getElementById('chart');
  const leitura = document.getElementById('leitura');
  let ultimo = -1;

  function fmt(v) { return v == null ? '—' : v.toFixed(1) + '°'; }

  function atualizar() {
    const f = Math.max(0, Math.min(Math.round(vid.currentTime * fps), n - 1));
    if (f !== ultimo) {
      ultimo = f;
      Plotly.update('chart',
        {x: [[f], [f]], y: [[knee[f]], [hip[f]]]},
        {'shapes[0].x0': f, 'shapes[0].x1': f},
        [2, 3]);
      leitura.innerHTML = 'Quadro ' + f + ' · ' + (f / fps).toFixed(2) + ' s' +
        ' &nbsp;—&nbsp; <b>Joelho:</b> ' + fmt(knee[f]) +
        ' &nbsp;·&nbsp; <b>Quadril:</b> ' + fmt(hip[f]);
    }
    requestAnimationFrame(atualizar);
  }
  requestAnimationFrame(atualizar);

  // Clicar no gráfico leva o vídeo para o instante correspondente
  chart.on('plotly_click', ev => {
    const f = Math.round(ev.points[0].x);
    vid.currentTime = f / fps;
  });

  // Em telas estreitas (celular), empilha vídeo e gráfico verticalmente
  const ALTURA = ${altura};
  const ALTURA_MOBILE = 300;
  const wrap = document.getElementById('wrap');
  let estreitoAtual = null;

  function alturaTotal() {
    return (estreitoAtual ? 2 * ALTURA_MOBILE + 16 : ALTURA) + 55;
  }

  // O Streamlit reaplica a altura original do iframe; forçamos a nossa
  // via atributo e estilo com !important, reaplicados periodicamente.
  function fixarAlturaIframe() {
    const iframe = window.frameElement;
    if (!iframe || estreitoAtual === null) return;
    const total = alturaTotal();
    iframe.setAttribute('height', total);
    iframe.style.setProperty('height', total + 'px', 'important');
  }

  function ajustarLayout() {
    const estreito = window.innerWidth < 640;
    if (estreito === estreitoAtual) return;
    estreitoAtual = estreito;

    wrap.style.flexDirection = estreito ? 'column' : 'row';
    const h = estreito ? ALTURA_MOBILE : ALTURA;
    vid.style.height = h + 'px';
    chart.style.height = h + 'px';

    fixarAlturaIframe();
    Plotly.Plots.resize(chart);
  }

  window.addEventListener('resize', ajustarLayout);
  ajustarLayout();
  setInterval(fixarAlturaIframe, 500);
</script>
"""
)


def player_sincronizado(video_bytes, df, fps, altura=430):
    """Renderiza vídeo (com play) e gráfico Plotly sincronizados lado a lado."""

    def serie_json(coluna):
        return json.dumps(
            [None if pd.isna(v) else round(float(v), 2) for v in df[coluna]]
        )

    html = _TEMPLATE_PLAYER.substitute(
        altura=altura,
        video_b64=base64.b64encode(video_bytes).decode(),
        knee=serie_json("Knee_Angle"),
        hip=serie_json("Hip_Angle"),
        fps=f"{fps:.6f}",
    )
    components.html(html, height=altura + 55)


# ----------------------------------------------------------------------------
# Métricas do agachamento
# ----------------------------------------------------------------------------


def calcular_metricas(df, fps):
    """Tabela de métricas por articulação: picos, amplitude e velocidade angular."""
    valores = {}
    for nome, coluna in (("Joelho", "Knee_Angle"), ("Quadril", "Hip_Angle")):
        serie = df[coluna]
        interp = serie.interpolate(limit_direction="both").to_numpy()
        vel = np.gradient(interp, 1.0 / fps)  # graus por segundo

        valores[nome] = {
            "Flexão máxima (°)": f"{serie.max():.1f}",
            "Instante da flexão máxima (s)": f"{serie.idxmax() / fps:.2f}",
            "Flexão mínima (°)": f"{serie.min():.1f}",
            "Amplitude de movimento (°)": f"{serie.max() - serie.min():.1f}",
            "Pico de velocidade angular — flexão (°/s)": f"{np.nanmax(vel):.1f}",
            "Pico de velocidade angular — extensão (°/s)": f"{abs(np.nanmin(vel)):.1f}",
            "Velocidade angular média (°/s)": f"{np.nanmean(np.abs(vel)):.1f}",
        }

    metricas = list(valores["Joelho"].keys())
    return pd.DataFrame(
        {
            "Métrica": metricas,
            "Joelho": [valores["Joelho"][m] for m in metricas],
            "Quadril": [valores["Quadril"][m] for m in metricas],
        }
    )


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------

st.markdown('<div class="app-title">Análise Markerless dos Membros Inferiores (ex: agachamento)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Cinemática 2D de quadril e joelho a partir de vídeo '
    "na vista sagital, sem marcadores — MediaPipe Pose.</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">1 · Envio do vídeo</div>', unsafe_allow_html=True)

col_up, col_cfg = st.columns([2, 1], gap="large")

PASTA_EXEMPLOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")

with col_up:
    arquivo = st.file_uploader(
        "Vídeo do agachamento (vista lateral / plano sagital)",
        type=["mp4", "mov", "avi", "mkv"],
        help="Grave o movimento perpendicular ao plano sagital, com quadril, joelho e tornozelo visíveis.",
    )

    exemplos = (
        sorted(
            f for f in os.listdir(PASTA_EXEMPLOS)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        )
        if os.path.isdir(PASTA_EXEMPLOS)
        else []
    )
    exemplo = None
    if arquivo is None and exemplos:
        escolha = st.selectbox(
            "Ou escolha um vídeo de exemplo",
            ["—"] + exemplos,
            help="Vídeos de demonstração incluídos no projeto (pasta videos/).",
        )
        if escolha != "—":
            exemplo = os.path.join(PASTA_EXEMPLOS, escolha)

with col_cfg:
    lado = st.radio(
        "Lado analisado",
        options=["Direito", "Esquerdo"],
        horizontal=True,
        help="Lado do corpo voltado para a câmera.",
    )
    with st.expander("Parâmetros avançados"):
        min_confidence = st.slider(
            "Confiança mínima (MediaPipe)", 0.1, 1.0, 0.7, 0.05,
            help="Limiar de detecção e rastreamento dos marcadores virtuais.",
        )

side = "r" if lado == "Direito" else "l"

if arquivo is not None:
    nome_video = arquivo.name
    tamanho_video = arquivo.size
elif exemplo is not None:
    nome_video = os.path.basename(exemplo)
    tamanho_video = os.path.getsize(exemplo)
else:
    nome_video = None

if nome_video is not None:
    chave = (nome_video, tamanho_video, side, min_confidence)
    ja_processado = st.session_state.get("chave") == chave

    if not ja_processado:
        if st.button("Processar vídeo", type="primary"):
            if arquivo is not None:
                sufixo = os.path.splitext(arquivo.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
                    tmp.write(arquivo.getbuffer())
                    caminho_video = tmp.name
                temporario = True
            else:
                caminho_video = exemplo
                temporario = False

            barra = st.progress(0.0, text="Processando o vídeo com MediaPipe Pose…")
            try:
                df, video_out, fps = processar_video(
                    caminho_video,
                    side=side,
                    min_confidence=min_confidence,
                    progress_callback=lambda p: barra.progress(p, text=f"Processando… {p:.0%}"),
                )
            finally:
                if temporario:
                    os.unlink(caminho_video)
            barra.empty()

            if df["Knee_Angle"].dropna().empty:
                st.error(
                    "Nenhuma pose foi detectada no vídeo. Verifique se o corpo inteiro "
                    "está visível na vista sagital ou reduza a confiança mínima."
                )
                st.stop()

            with open(video_out, "rb") as f:
                video_bytes = f.read()

            st.session_state["chave"] = chave
            st.session_state["df"] = df
            st.session_state["video_bytes"] = video_bytes
            st.session_state["fps"] = fps
            st.session_state["nome_base"] = os.path.splitext(nome_video)[0]
            st.rerun()
    else:
        df = st.session_state["df"]
        video_bytes = st.session_state["video_bytes"]
        fps = st.session_state["fps"]
        nome_base = st.session_state["nome_base"]

        # ------------------------------------------------------------------
        # 2. Análise sincronizada (vídeo + gráfico)
        # ------------------------------------------------------------------
        st.markdown(
            '<div class="section-label">2 · Análise sincronizada</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Dê play no vídeo ou arraste a barra de tempo: o gráfico acompanha o "
            "movimento em tempo real. Clique em um ponto do gráfico para levar o "
            "vídeo ao instante correspondente."
        )
        player_sincronizado(video_bytes, df, fps)

        # ------------------------------------------------------------------
        # 3. Métricas do agachamento
        # ------------------------------------------------------------------
        st.markdown(
            '<div class="section-label">3 · Métricas do agachamento</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            calcular_metricas(df, fps),
            hide_index=True,
            use_container_width=True,
        )
        duracao = len(df) / fps
        taxa = df["Knee_Angle"].notna().mean()
        st.caption(
            f"{len(df)} quadros · {fps:.2f} fps · {duracao:.2f} s · "
            f"pose detectada em {taxa:.0%} dos quadros. Velocidades calculadas por "
            "diferenciação numérica dos ângulos (quadros sem detecção são interpolados)."
        )

        # ------------------------------------------------------------------
        # 4. Exportar resultados
        # ------------------------------------------------------------------
        st.markdown('<div class="section-label">4 · Exportar resultados</div>', unsafe_allow_html=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        col_d1, col_d2, _ = st.columns([1, 1, 2])
        with col_d1:
            st.download_button(
                "Baixar CSV",
                data=csv_bytes,
                file_name=f"{nome_base}_markerless_{side}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                "Baixar vídeo",
                data=video_bytes,
                file_name=f"{nome_base}_markerless_{side}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
else:
    st.info(
        "Envie um vídeo do agachamento gravado na vista lateral (plano sagital) "
        "para iniciar a análise."
    )

st.markdown(
    """
    <div class="citacao-box">
        <div class="citacao-titulo">Como citar</div>
        <div class="citacao-texto">
            PEREIRA, Dayanne R.; CATELLI, Danilo S.; SANTIAGO, Paulo R. P.; BEDO, Bruno L. S.
            <b>Markerless Pixel-Based Pipeline for Quantifying 2D Lower Limb Kinematics During
            Squatting: A Preliminary Validation Study</b>. <i>Biomechanics</i>, v. 6, n. 1, p. 1,
            2025. MDPI. DOI: 10.3390/biomechanics6010001.
        </div>
        <div class="citacao-links">
            <a class="citacao-link" href="https://doi.org/10.3390/biomechanics6010001"
               target="_blank" rel="noopener">Acessar o artigo ↗</a>
            <a class="citacao-link secundario" href="https://github.com/brunobedo/mediapipe2dangle"
               target="_blank" rel="noopener">Repositório no GitHub ↗</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
