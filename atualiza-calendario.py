import os
import re

html_calendario = 'ql-measurement-calendar.html' 
cor_padrao = '#A3E4D7'              

with open(html_calendario, 'r', encoding='utf-8') as f:
    conteudo = f.read()

# Pega URLs já cadastradas para não duplicar
urls_existentes = set(re.findall(r"url:\s*'(.*?)'", conteudo))

novas_entradas = ""
contador = 0

# Varre a pasta atual procurando pastas de anos (ex: "2024")
for pasta in os.listdir('.'):
    if os.path.isdir(pasta) and pasta.isdigit() and len(pasta) == 4:
        for arquivo in os.listdir(pasta):
            if arquivo.endswith('_Gallery.html'):
                url_relativa = f"{pasta}/{arquivo}"
                
                if url_relativa not in urls_existentes:
                    # Extrai os primeiros 8 números do arquivo (YYYYMMDD)
                    match = re.match(r'^(\d{4})(\d{2})(\d{2})', arquivo)
                    if match:
                        ano = int(match.group(1))
                        mes_js = int(match.group(2)) - 1 # JS conta meses de 0 a 11
                        dia = int(match.group(3))
                        
                        novas_entradas += f"  {{\n    startDate: new Date({ano}, {mes_js}, {dia}), endDate: new Date({ano}, {mes_js}, {dia}), color: '{cor_padrao}', url: '{url_relativa}'  }},\n"
                        contador += 1

if contador == 0:
    print("Nenhuma medida nova para adicionar.")
else:
    print(f"Adicionando {contador} novas medidas no calendário...")
    if "// MARCADOR_AUTOMATICO" in conteudo:
        novo_conteudo = conteudo.replace("// MARCADOR_AUTOMATICO", novas_entradas + "  // MARCADOR_AUTOMATICO")
        with open(html_calendario, 'w', encoding='utf-8') as f:
            f.write(novo_conteudo)
        print("Calendário atualizado com sucesso!")
    else:
        print("ERRO: Faltou colocar o // MARCADOR_AUTOMATICO no seu HTML.")
