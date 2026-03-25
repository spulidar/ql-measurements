"""
LIdar IMage Publisher - LIMP
Script para coletar gráficos (.webp), gerar Dashboards HTML planos por ano,
e atualizar automaticamente o calendário interativo de medidas.
"""

import os
import glob
import shutil
import re
from datetime import datetime

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
INCREMENTAL_PROCESSING = True  

rootdir_name = os.getcwd() 
files_dir_level1 = "03-data_level1"  
site_dir = "ql-measurements"          
calendar_file = "ql-measurement-calendar.html" 

base_data_folder = os.path.join(rootdir_name, files_dir_level1)
base_site_folder = os.path.join(rootdir_name, site_dir) 

# ==========================================
# GERAÇÃO DO DASHBOARD HTML
# ==========================================
def generate_html_dashboard(html_path, prefix, date_title, valid_channels, valid_alts, has_global_mean, mean_rcs_file):
    default_ch = valid_channels[0] if valid_channels else ""
    default_alt = valid_alts[0] if valid_alts else ""
    
    channel_buttons = ""
    for ch in valid_channels:
        pretty_name = ch.replace("_", " ") 
        channel_buttons += f'<button class="tab-btn ch-btn" onclick="setChannel(\'{ch}\', this)">{pretty_name}</button>\n'

    altitude_buttons = ""
    for alt in valid_alts:
        altitude_buttons += f'<button class="tab-btn alt-btn" onclick="setAltitude(\'{alt}\', this)">{alt} km</button>\n'

    global_tab_style = "display: inline-block;" if has_global_mean else "display: none;"

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Lidar Measurements - {date_title}</title>
  <style type="text/css">
    html, body {{ background: #f4f4f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; text-align: center; }}
    h2 {{ color: #333; margin-top: 20px; }}
    .subtitle {{ font-size: 18px; font-weight: normal; color: #666; display: block; margin-top: 5px; }}
    .dashboard {{ max-width: 100%; margin: 0 auto; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
    
    .main-tabs {{ margin-bottom: 25px; }}
    .main-tab-btn {{ background: transparent; border: none; font-size: 18px; font-weight: bold; color: #777; cursor: pointer; padding: 10px 20px; margin: 0 10px; border-bottom: 3px solid transparent; transition: color 0.3s; }}
    .main-tab-btn:hover {{ color: #0056b3; }}
    .main-tab-btn.active {{ color: #0056b3; border-bottom: 3px solid #0056b3; }}
    
    .controls {{ display: flex; justify-content: center; gap: 40px; margin-bottom: 25px; flex-wrap: wrap; }}
    .tab-group {{ background: white; padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    .tab-group h3 {{ margin: 0 0 15px 0; font-size: 16px; color: #555; text-transform: uppercase; letter-spacing: 1px; }}
    .tab-btn {{ background: #e0e0e0; border: none; padding: 10px 20px; margin: 0 5px; border-radius: 5px; cursor: pointer; font-size: 15px; font-weight: bold; color: #333; transition: background 0.2s, transform 0.1s; }}
    .tab-btn:hover {{ background: #d0d0d0; transform: translateY(-2px); }}
    .tab-btn.active {{ background: #0056b3; color: white; box-shadow: 0 4px 8px rgba(0,86,179,0.3); }}
    
    .image-display {{ display: flex; flex-direction: column; align-items: center; gap: 20px; width: 100%; }}
    
    /* A CAIXA DA IMAGEM AGORA TEM TRANSIÇÃO SUAVE DE TAMANHO */
    .image-card {{ 
        background: white; padding: 15px; border-radius: 8px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.15); 
        width: 100%; 
        max-width: 80%; /* Inicia em 80% para o Quicklook */
        transition: max-width 0.4s ease-in-out; 
    }}
    .image-card img {{ width: 100%; height: auto; cursor: zoom-in; border-radius: 4px; display: block; }}
    
    #myModal {{ display: none; position: fixed; z-index: 100; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.85); }}
    .modal-content {{ margin: auto; display: block; width: 95%; max-width: 1600px; margin-top: 2%; animation: zoom 0.3s ease-in-out; }}
    @keyframes zoom {{ from {{transform:scale(0)}} to {{transform:scale(1)}} }}
  </style>
</head>
<body>
  <h2>Lidar Measurements<br><span class="subtitle">{date_title}</span></h2>
  
  <div class="dashboard">
      <div class="main-tabs">
          <button class="main-tab-btn active" id="tab-quicklooks" onclick="setMode('quicklooks')">Quicklooks</button>
          <button class="main-tab-btn" id="tab-resumo" style="{global_tab_style}" onclick="setMode('resumo')">Resumo Global</button>
      </div>

      <div id="controls-panel" class="controls">
          <div class="tab-group">
              <h3>CHANNEL</h3>
              {channel_buttons}
          </div>
          <div class="tab-group">
              <h3>ALTITUDE</h3>
              {altitude_buttons}
          </div>
      </div>

      <div class="image-display">
          <div class="image-card" id="img-card">
              <img id="main-display" src="Quicklook_{prefix}_{default_ch}_{default_alt}km.webp" onclick="openModal(this.src)" alt="Lidar Image">
          </div>
      </div>
  </div>

  <div id="myModal" onclick="closeModal()">
    <img class="modal-content" id="img01">
  </div>

  <script>
    var currentChannel = "{default_ch}";
    var currentAltitude = "{default_alt}";
    var prefix = "{prefix}";
    var currentMode = "quicklooks";

    document.addEventListener("DOMContentLoaded", function() {{
        var firstCh = document.querySelector(".ch-btn");
        var firstAlt = document.querySelector(".alt-btn");
        if(firstCh) firstCh.classList.add("active");
        if(firstAlt) firstAlt.classList.add("active");
    }});

    function updateImage() {{
        var imgElement = document.getElementById("main-display");
        if (currentMode === "quicklooks") {{
            imgElement.src = "Quicklook_" + prefix + "_" + currentChannel + "_" + currentAltitude + "km.webp";
        }} else {{
            imgElement.src = "{mean_rcs_file}";
        }}
    }}

    function setMode(mode) {{
        currentMode = mode;
        document.getElementById("tab-quicklooks").classList.remove("active");
        document.getElementById("tab-resumo").classList.remove("active");
        
        var imgCard = document.getElementById("img-card"); // Pega a caixa da imagem

        if (mode === "quicklooks") {{
            document.getElementById("tab-quicklooks").classList.add("active");
            document.getElementById("controls-panel").style.display = "flex"; 
            imgCard.style.maxWidth = "80%"; // Define para 80% da tela
        }} else {{
            document.getElementById("tab-resumo").classList.add("active");
            document.getElementById("controls-panel").style.display = "none"; 
            imgCard.style.maxWidth = "50%"; // Encolhe para 50% da tela
        }}
        updateImage();
    }}

    function setChannel(ch, btnElement) {{
        currentChannel = ch;
        document.querySelectorAll(".ch-btn").forEach(btn => btn.classList.remove("active"));
        btnElement.classList.add("active");
        updateImage();
    }}

    function setAltitude(alt, btnElement) {{
        currentAltitude = alt;
        document.querySelectorAll(".alt-btn").forEach(btn => btn.classList.remove("active"));
        btnElement.classList.add("active");
        updateImage();
    }}

    var modal = document.getElementById("myModal");
    var modalImg = document.getElementById("img01");
    function openModal(src) {{ modal.style.display = "block"; modalImg.src = src; }}
    function closeModal() {{ modal.style.display = "none"; }}
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ==========================================
# INTEGRAÇÃO DO CALENDÁRIO
# ==========================================
def update_calendar():
    print(f"\n[INFO] LIMP: Sincronizando o calendário ({calendar_file})...")
    
    caminho_calendario = os.path.join(base_site_folder, calendar_file)
    cor_padrao = '#A3E4D7'
    
    if not os.path.exists(caminho_calendario):
        print(f"  -> [ERRO] O arquivo {calendar_file} não foi encontrado na pasta raiz!")
        return

    with open(caminho_calendario, 'r', encoding='utf-8') as f:
        conteudo = f.read()

    # Pega URLs já cadastradas para não duplicar
    urls_existentes = set(re.findall(r"url:\s*'(.*?)'", conteudo))

    novas_entradas = ""
    contador = 0

    # Varre a pasta atualiza-site procurando anos (ex: "2024")
    for pasta_ano in os.listdir(base_site_folder):
        caminho_ano = os.path.join(base_site_folder, pasta_ano)
        
        if os.path.isdir(caminho_ano) and pasta_ano.isdigit() and len(pasta_ano) == 4:
            for arquivo in os.listdir(caminho_ano):
                # Aceita tanto os Dashboards novos quanto as Gallery antigas
                if arquivo.endswith('_Dashboard.html') or arquivo.endswith('_Gallery.html'):
                    url_relativa = f"{pasta_ano}/{arquivo}"
                    
                    if url_relativa not in urls_existentes:
                        # Extrai os primeiros 8 números do arquivo (YYYYMMDD)
                        match = re.match(r'^(\d{4})(\d{2})(\d{2})', arquivo)
                        if match:
                            ano = int(match.group(1))
                            mes_js = int(match.group(2)) - 1 # JS conta meses de 0 a 11
                            dia = int(match.group(3))
                            
                            novas_entradas += f"  {{\n    startDate: new Date({ano}, {mes_js}, {dia}), endDate: new Date({ano}, {mes_js}, {dia}), color: '{cor_padrao}', url: '{url_relativa}'\n  }},\n"
                            contador += 1

    if contador == 0:
        print("  -> O calendário já está atualizado. Nenhuma nova medida adicionada.")
    else:
        print(f"  -> Inserindo {contador} novas medidas no calendário...")
        if "// MARCADOR_AUTOMATICO" in conteudo:
            novo_conteudo = conteudo.replace("// MARCADOR_AUTOMATICO", novas_entradas + "  // MARCADOR_AUTOMATICO")
            with open(caminho_calendario, 'w', encoding='utf-8') as f:
                f.write(novo_conteudo)
            print("  -> [OK] Calendário sincronizado com sucesso!")
        else:
            print("  -> [ERRO] Faltou colocar o // MARCADOR_AUTOMATICO no seu HTML.")


# ==========================================
# ROTINA PRINCIPAL DE PUBLICAÇÃO (O "ASPIRADOR")
# ==========================================
def run_limp():
    print(f"[INFO] LIMP: Iniciando a varredura e publicação HTML...")
    os.makedirs(base_site_folder, exist_ok=True)
    
    search_pattern = os.path.join(base_data_folder, '**', '*.webp')
    all_images = glob.glob(search_pattern, recursive=True)
    
    if not all_images:
        print(f"[AVISO] Nenhuma imagem '.webp' encontrada em {files_dir_level1}.")
        return

    measurements = {}
    
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        prefix = None
        
        if img_name.startswith("Quicklook_"):
            parts = img_name.replace(".webp", "").split("_")
            if len(parts) >= 5:
                prefix = parts[1]
                ch = f"{parts[2]}_{parts[3]}" 
                alt = parts[4].replace("km", "")
        elif img_name.startswith("GlobalMeanRCS_"):
            parts = img_name.replace(".webp", "").split("_")
            if len(parts) >= 2:
                prefix = parts[1]
            
        if prefix:
            if prefix not in measurements:
                measurements[prefix] = {
                    'files': [], 'channels': set(), 'alts': set(), 'has_global_mean': False, 'mean_rcs_filename': ""
                }
            
            measurements[prefix]['files'].append(img_path)
            
            if img_name.startswith("Quicklook_"):
                measurements[prefix]['channels'].add(ch)
                measurements[prefix]['alts'].add(alt)
            elif img_name.startswith("GlobalMeanRCS_"):
                measurements[prefix]['has_global_mean'] = True
                measurements[prefix]['mean_rcs_filename'] = img_name

    dias_processados = 0

    for prefix, data in measurements.items():
        try:
            year = prefix[:4]
            dt = datetime.strptime(prefix[:8], "%Y%m%d")
            date_str = dt.strftime("%d %b %Y")
        except:
            year = "Unknown"
            date_str = prefix

        site_year_folder = os.path.join(base_site_folder, year)
        os.makedirs(site_year_folder, exist_ok=True)
        
        html_path = os.path.join(site_year_folder, f"{prefix}_Dashboard.html")

        if INCREMENTAL_PROCESSING and os.path.exists(html_path):
            print(f"  -> [PULADO] Dashboard já existe para: {prefix}")
            continue
            
        for img_path in data['files']:
            shutil.copy2(img_path, os.path.join(site_year_folder, os.path.basename(img_path)))
            
        valid_channels = sorted(list(data['channels']))
        valid_alts = sorted(list(data['alts']), key=lambda x: float(x) if x.replace('.','',1).isdigit() else 0)
        
        if valid_channels:
            generate_html_dashboard(
                html_path, 
                prefix, 
                date_str, 
                valid_channels, 
                valid_alts, 
                data['has_global_mean'], 
                data['mean_rcs_filename']
            )
            print(f"  -> [OK] Publicado no site: {prefix}")
            dias_processados += 1

    print(f"\n[INFO] LIMP: Geração HTML finalizada! {dias_processados} novos painéis na pasta '{site_dir}'.")
    
    # Chama a função de atualização do calendário assim que terminar os HTMLs
    update_calendar()

if __name__ == "__main__":
    run_limp()