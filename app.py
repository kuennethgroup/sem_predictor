import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import json
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import pickle
from pathlib import Path

# --- SEITENKONFIGURATION & DESIGN ---
st.set_page_config(
    page_title="🔬 SEM Phasen-Analyse",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# Custom CSS für ein professionelleres Aussehen
st.markdown("""
<style>
    /* Haupt-Container anpassen, um den Abstand oben zu reduzieren */
    .block-container {
        padding-top: 2rem;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.75rem;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .instruction-box {
        padding: 1rem; 
        border-radius: 10px; 
        height: 100%;
    }
    .instruction-box h3 {
        font-size: 1.5rem;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    .stButton>button {
        border-radius: 10px;
        border: 2px solid #1f77b4;
        background-color: #ffffff;
        color: #1f77b4;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1f77b4;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- ÜBERSETZUNGEN ---
TRANSLATIONS = {
    "de": {
        "main_header": "Analyse von BSE-Bildern für Cr-Si-Legierungen",
        "welcome_message": "Willkommen! Dieses Werkzeug nutzt Machine-Learning-Modelle, um primär die Phasenanteile in BSE-Bildern schnell und präzise zu bestimmen. Optional ist das Modell auch in der Lage die Legierungszusammensetzungen zu bestimmen.",
        "config_header": "⚙️ Konfiguration",
        "language_select_label": "Sprache auswählen:",
        "model_select_label": "1. Wählen Sie ein Analyse-Modell:",
        "model_loaded_success": "Modell:** '{model_name}' geladen.",
        "model_loaded_error": "Modell konnte nicht geladen werden. Bitte überprüfen Sie die Pfade.",
        "phase_select_label": "2. Wählen Sie die zu analysierenden Phasen oder ein Element aus:",
        "upload_header": "📁 Bild-Upload",
        "upload_label": "Laden Sie ein oder mehrere Bilder hoch:",
        "instruction_header": "So funktioniert's:",
        "instruction_step1_title": "Modell wählen:",
        "instruction_step1_text": "Wählen Sie links in der Konfiguration ein trainiertes Analyse-Modell.",
        "instruction_step2_title": "Bilder hochladen:",
        "instruction_step2_text": "Laden Sie ein oder mehrere BSE-Bilder per Klick oder Drag-and-Drop hoch.",
        "instruction_step3_title": "Bilder auswählen:",
        "instruction_step3_text": "Markieren Sie im Hauptbereich die Bilder, die Sie analysieren möchten. Nutzen Sie den \"Alle auswählen\"-Button für eine schnellere Auswahl.",
        "instruction_step4_title": "Analyse starten:",
        "instruction_step4_text": "Klicken Sie auf den \"Vorhersage starten\"-Button.",
        "instruction_step5_title": "Ergebnisse:",
        "instruction_step5_text": "Die Ergebnisse werden als Tabelle mit Fehlerangaben angezeigt und können als CSV-Datei heruntergeladen werden.",
        "main_content_info": "Bitte laden Sie Bilder über die Seitenleiste links hoch, um mit der Analyse zu beginnen.",
        "image_selection_header": "🖼️ Bildauswahl",
        "select_all_button": "Alle auswählen",
        "deselect_all_button": "Alle abwählen",
        "checkbox_label": "Auswählen",
        "warning_no_image_selected": "Bitte wählen Sie mindestens ein Bild für die Analyse aus.",
        "warning_no_phase_selected": "Bitte wählen Sie mindestens eine Phase in der Seitenleiste aus.",
        "predict_button_label": "🚀 Vorhersage für {count} Bild(er) starten",
        "results_header": "📊 Ergebnisse",
        "progress_bar_start": "Analyse startet...",
        "progress_bar_analyzing": "Analysiere: {filename}",
        "download_csv_button": "📥 Ergebnisse als CSV herunterladen",
        "analysis_error": "Die Analyse konnte für die ausgewählten Bilder keine Ergebnisse liefern.",
        "footer_text": "Entwickelt im Rahmen einer Masterarbeit im Bereich Computational Materials Science."
    },
    "en": {
        "main_header": "Analysis of BSE Images for Cr-Si Alloys",
        "welcome_message": "Welcome! This tool uses machine learning models to quickly and accurately determine the phase fractions in BSE images. Optionally the model can also determine the alloy compositions.",
        "config_header": "⚙️ Configuration",
        "language_select_label": "Select Language:",
        "model_select_label": "1. Select an analysis model:",
        "model_loaded_success": "Model:** '{model_name}' loaded.",
        "model_loaded_error": "Could not load the model. Please check the paths.",
        "phase_select_label": "2. Select the phases to analyze or an element:",
        "upload_header": "📁 Image Upload",
        "upload_label": "Upload one or more images:",
        "instruction_header": "How it works:",
        "instruction_step1_title": "Select Model:",
        "instruction_step1_text": "In the configuration on the left, select a trained analysis model.",
        "instruction_step2_title": "Upload Images:",
        "instruction_step2_text": "Upload one or more BSE images by clicking or dragging and dropping.",
        "instruction_step3_title": "Select Images:",
        "instruction_step3_text": "In the main area, check the images you want to analyze. Use the \"Select All\" button for faster selection.",
        "instruction_step4_title": "Start Analysis:",
        "instruction_step4_text": "Click the \"Start Prediction\" button.",
        "instruction_step5_title": "Results:",
        "instruction_step5_text": "The results are displayed as a table with error margins and can be downloaded as a CSV file.",
        "main_content_info": "Please upload images via the sidebar on the left to begin the analysis.",
        "image_selection_header": "🖼️ Image Selection",
        "select_all_button": "Select All",
        "deselect_all_button": "Deselect All",
        "checkbox_label": "Select",
        "warning_no_image_selected": "Please select at least one image for analysis.",
        "warning_no_phase_selected": "Please select at least one phase in the sidebar.",
        "predict_button_label": "🚀 Start Prediction for {count} image(s)",
        "results_header": "📊 Results",
        "progress_bar_start": "Analysis starting...",
        "progress_bar_analyzing": "Analyzing: {filename}",
        "download_csv_button": "📥 Download results as CSV",
        "analysis_error": "The analysis could not produce results for the selected images.",
        "footer_text": "Developed as part of a Master's thesis in Computational Materials Science."
    }
}

if 'language' not in st.session_state:
    st.session_state['language'] = 'de'

def get_translation(key):
    return TRANSLATIONS[st.session_state['language']].get(key, key)


# --- MODELL- UND PREDICTOR-DEFINITIONEN ---
class SEM_Model_ResNet50(nn.Module):
    def __init__(self, num_fc_layers, fc_hidden_sizes, dropout_rates, selector_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for param in resnet.parameters():
            param.requires_grad = False
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_size = 2048
        self.fc_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        input_size = self.feature_size + selector_size
        for i in range(num_fc_layers):
            output_size = fc_hidden_sizes[i] if i < len(fc_hidden_sizes) else 1
            self.fc_layers.append(nn.Linear(input_size, output_size))
            if i < len(self.fc_layers) - 1:
                dropout_rate = dropout_rates[i] if i < len(dropout_rates) else dropout_rates[-1]
                self.dropout_layers.append(nn.Dropout(dropout_rate))
            input_size = output_size
    def forward(self, img, selector):
        features = self.feature_extractor(img)
        features = features.view(features.size(0), -1)
        x = torch.cat((features, selector), dim=1)
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if i < len(self.fc_layers) - 1:
                x = torch.nn.functional.relu(x)
                if i < len(self.dropout_layers):
                    x = self.dropout_layers[i](x)
        return x

class SEMPredictor:
    def __init__(self, model_config, device):
        self.device = device
        self.compositions = model_config["compositions"]
        self.model = model_config["model"]
        self.scalers = model_config["scalers"]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
    def predict(self, image_file, selected_compositions):
        results = {}
        try:
            image = Image.open(image_file).convert('L')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            for composition in selected_compositions:
                if composition not in self.compositions:
                    continue
                selector = torch.zeros(1, len(self.compositions), device=self.device)
                selector[0, self.compositions.index(composition)] = 1
                with torch.no_grad():
                    scaled_prediction = self.model(image_tensor, selector)
                    scaled_value = scaled_prediction.cpu().numpy()
                original_value = self.scalers[composition].inverse_transform(scaled_value)[0][0]
                results[composition] = float(original_value)
        except Exception as e:
            st.error(f"Fehler bei der Verarbeitung von {image_file.name}: {e}")
            return None
        return results
    
@st.cache_resource
def load_model_and_dependencies(model_name, _config):
    try:
        # Konvertiere Path-Objekte zu Strings für Dateioperationen
        model_path = str(_config["model_path"])
        params_path = str(_config["params_path"])
        scaler_path = str(_config["scaler_path"])

        with open(params_path, 'r') as f:
            params = json.load(f)
        fc_hidden_sizes = _config["architecture_builder"](params)
        dropout_rates = [params['dropout1'], params.get('dropout2', 0.5)]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SEM_Model_ResNet50(
            num_fc_layers=params['num_fc_layers'], fc_hidden_sizes=fc_hidden_sizes,
            dropout_rates=dropout_rates, selector_size=len(_config["compositions"])
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        return {"model": model, "scalers": scalers, "compositions": _config["compositions"], "device": device}
    except Exception as e:
        st.error(f"Kritisches Problem beim Laden von '{model_name}': {e}")
        return None


# Nutze den Ordner, in dem die app.py liegt, als Basis-Pfad
BASE_PATH = Path(__file__).parent.resolve()

MODEL_CONFIGS = {
    "Modell 1: ResNet50 (mit Platin)": {
        "model_path": BASE_PATH / 'models/ResNet50_ALL.pth',
        "params_path": BASE_PATH / 'models/ResNet50_ALL_parameters.json',
        "scaler_path": BASE_PATH / 'models/scalers_ALL.pkl',
        "compositions": ['A15 Phase', 'Pores','Chromium', 'Silicon', 'Germanium', 'Molybdenum', 'Platinum', 'CrMK'],
        "architecture_builder": lambda p: [p['hidden1'], p['hidden2'], p['hidden2'], p['hidden2']],
        "std_devs": {
            'A15 Phase': (3.9, 'F.-%'), 'Chromium': (2.4, 'At.-%'), 'Silicon': (1.5, 'At.-%'), 
            'Germanium': (0.8, 'At.-%'), 'Molybdenum': (1.3, 'At.-%'), 'Platinum': (1.0, 'At.-%'), 
            'Pores': (0.5, 'F.-%'), 'CrMK': (4.9, 'F.-%')
        }
    },
    "Modell 2: ResNet50 (ohne Platin)": {
        "model_path": BASE_PATH / 'models/ResNet50_noPt.pth',
        "params_path": BASE_PATH / 'models/ResNet50_noPt_parameters.json',
        "scaler_path": BASE_PATH / 'models/scalers_noPt.pkl',
        "compositions": ['A15 Phase', 'Pores','Chromium', 'Silicon', 'Germanium', 'Molybdenum', 'CrMK'],
        "architecture_builder": lambda p: [p['hidden1'], p['hidden2']],
        "std_devs": {
            'A15 Phase': (5.8, 'F.-%'), 'Chromium': (3.5, 'At.-%'), 'Silicon': (0.9, 'At.-%'), 
            'Germanium': (1.2, 'At.-%'), 'Molybdenum': (2.8, 'At.-%'), 'Pores': (0.3, 'F.-%'), 
            'CrMK': (5.9, 'F.-%')
        }
    }
}




# --- STREAMLIT OBERFLÄCHE ---
with st.sidebar:
    st.header(get_translation("config_header"))
    lang_options = {"Deutsch": "de", "English": "en"}
    lang_choice = st.radio(
        get_translation("language_select_label"), options=lang_options.keys(), horizontal=True,
        index=list(lang_options.values()).index(st.session_state.language)
    )
    st.session_state.language = lang_options[lang_choice]
    st.markdown("---")
    selected_model_name = st.selectbox(get_translation("model_select_label"), list(MODEL_CONFIGS.keys()))
    active_config = MODEL_CONFIGS[selected_model_name]
    loaded_model_data = load_model_and_dependencies(selected_model_name, active_config)
    if loaded_model_data:
        st.success(get_translation("model_loaded_success").format(model_name=selected_model_name))
        available_compositions = loaded_model_data["compositions"]
        selected_compositions = st.multiselect(
            get_translation("phase_select_label"), available_compositions, default=available_compositions
        )
    else:
        st.error(get_translation("model_loaded_error"))
        st.stop()
    st.header(get_translation("upload_header"))
    uploaded_files = st.file_uploader(
        get_translation("upload_label"), type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], accept_multiple_files=True
    )

main_col, instruction_col = st.columns([2.5, 1], gap="large")

with instruction_col:
    st.markdown(f"""
    <div class="instruction-box">
        <h3>{get_translation("instruction_header")}</h3>
        <ol style='padding-left: 25px; margin-bottom: 0px;'>
            <li><strong>{get_translation("instruction_step1_title")}</strong><br>{get_translation("instruction_step1_text")}</li><br>
            <li><strong>{get_translation("instruction_step2_title")}</strong><br>{get_translation("instruction_step2_text")}</li><br>
            <li><strong>{get_translation("instruction_step3_title")}</strong><br>{get_translation("instruction_step3_text")}</li><br>
            <li><strong>{get_translation("instruction_step4_title")}</strong><br>{get_translation("instruction_step4_text")}</li><br>
            <li><strong>{get_translation("instruction_step5_title")}</strong><br>{get_translation("instruction_step5_text")}</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with main_col:
    st.markdown(f'<h1 class="main-header">{get_translation("main_header")}</h1>', unsafe_allow_html=True)
    st.markdown(get_translation("welcome_message"))
    
    if not uploaded_files:
        st.info(get_translation("main_content_info"))
    else:
        st.markdown(f'<h2 class="sub-header">{get_translation("image_selection_header")}</h2>', unsafe_allow_html=True)
        if 'selected_images' not in st.session_state:
            st.session_state.selected_images = {}
        
        button_col, _, _, _ = st.columns(4)
        all_selected = all(st.session_state.selected_images.get(f.name, False) for f in uploaded_files) if uploaded_files else False
        def toggle_all_images(select_all):
            for f in uploaded_files:
                st.session_state.selected_images[f.name] = select_all
        
        with button_col:
            if all_selected:
                st.button(get_translation("deselect_all_button"), on_click=toggle_all_images, args=(False,), use_container_width=True)
            else:
                st.button(get_translation("select_all_button"), on_click=toggle_all_images, args=(True,), use_container_width=True)
        st.write("")
        
        cols = st.columns(4)
        for i, file in enumerate(uploaded_files):
            with cols[i % 4]:
                st.image(file, caption=file.name, use_container_width=True)
                is_selected = st.checkbox(get_translation("checkbox_label"), key=f"cb_{file.name}", value=st.session_state.selected_images.get(file.name, False))
                st.session_state.selected_images[file.name] = is_selected
        
        images_to_predict = [f for f in uploaded_files if st.session_state.selected_images.get(f.name)]
        st.markdown("---")
        
        if not images_to_predict:
            st.warning(get_translation("warning_no_image_selected"))
        elif not selected_compositions:
            st.warning(get_translation("warning_no_phase_selected"))
        else:
            if st.button(get_translation("predict_button_label").format(count=len(images_to_predict)), use_container_width=True):
                st.markdown(f'<h2 class="sub-header">{get_translation("results_header")}</h2>', unsafe_allow_html=True)
                predictor = SEMPredictor(loaded_model_data, loaded_model_data["device"])
                all_results = []
                progress_bar = st.progress(0, text=get_translation("progress_bar_start"))
                for i, image_file in enumerate(images_to_predict):
                    progress_bar.progress((i + 1) / len(images_to_predict), text=get_translation("progress_bar_analyzing").format(filename=image_file.name))
                    prediction = predictor.predict(image_file, selected_compositions)
                    if prediction:
                        prediction['Dateiname'] = image_file.name
                        all_results.append(prediction)
                progress_bar.empty()
                if all_results:
                    # --- KORRIGIERTE DATAFRAME-ERSTELLUNG UND -ANZEIGE ---
                    
                    # 1. Original-DataFrame für den Download erstellen
                    download_df = pd.DataFrame(all_results)
                    # Sicherstellen, dass Dateiname zuerst kommt
                    cols_for_download = ['Dateiname'] + [col for col in download_df if col != 'Dateiname']
                    download_df = download_df[cols_for_download]
                    
                    # 2. DataFrame für die Anzeige erstellen
                    display_df = download_df.copy()
                    std_devs = active_config.get("std_devs", {})
                    
                    # 3. Spaltennamen für die Anzeige formatieren
                    new_column_names = {}
                    for col in display_df.columns:
                        if col in std_devs:
                            error_val, unit = std_devs[col]
                            new_column_names[col] = f"{col}\n(±{error_val} {unit})"
                        else:
                            new_column_names[col] = col
                    
                    display_df.rename(columns=new_column_names, inplace=True)
                    
                    st.dataframe(
                        display_df.style.format("{:.4f}", subset=[col for col in display_df.columns if col != 'Dateiname']),
                        use_container_width=True
                    )
                    
                    # 4. Download-Button verwendet das unformatierte DataFrame
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv = convert_df_to_csv(download_df)
                    st.download_button(
                        label=get_translation("download_csv_button"), data=csv,
                        file_name=f"analyse_{selected_model_name.replace(' ', '_')}.csv", mime="text/csv",
                    )
                else:
                    st.error(get_translation("analysis_error"))
    
    st.markdown("<hr style='margin-top: 3rem;'>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #888;'>{get_translation('footer_text')}</p>", unsafe_allow_html=True)

