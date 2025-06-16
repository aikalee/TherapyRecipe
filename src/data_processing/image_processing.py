from bs4 import BeautifulSoup
import json
import re

def get_graph_metadata(graph, url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11351064/#"):

    

    figure = graph.find_parent("figure")
    figure_flag = False

    section = graph.find_parent(id=re.compile(r'^section\d+-\d+$'))
    section_id = section.get("id")
    section_url = url + section_id

    section_heading = section.find("h2").get_text()
    section_subheading = section.find("h3").get_text()
    headings = section_heading + " > " + section_subheading

    attribution = ""

    
    if figure:

        figure_flag = True

        image_url = graph.get("src")

        name = figure.select_one(".obj_head").get_text()
        all_p = [p.get_text() for p in figure.find_all("p") if not p.attrs]
        caption = all_p[0]
        label = name + " " + caption

        attribution = "(" + figure.select_one('[aria-label="Attribution"]').get_text() + ")"
        number = "_".join(re.findall(r"(.{1})\.", name)).lower()
        referee_id = f"figure_{number}"
        
          
    else:

        image_url = graph.get("src")

        table_section = graph.find_parent("section")

        name = table_section.select_one(".obj_head").get_text()
        caption = table_section.select_one(".caption p").get_text()
        label = name + " " + caption

        number = "_".join(re.findall(r"(.{1})\.", name)).lower()
        referee_id = f"table_{number}"
    

    return attribution, caption, figure_flag, headings, image_url, label, name, referee_id, section_url

def to_chunk(text_block, section_url, referee_id, headings):

    d = {
    "text": text_block,
    "metadata": {
        "section": section_url,
        "type": "table image",
        "referee_id": referee_id,
        "headings": headings,
        }
    }
    return d

def main():

    with open('../data/raw/source.html', encoding="utf-8") as f:
        html = f.read()
        soup = BeautifulSoup(html)

    with open("../data/processed/parsed_images.txt",  encoding="utf-8") as f:
        text = f.read()
        text_blocks = text.split("------")

    docs = []
    for graph, text_block in zip(soup.select(".graphic"), text_blocks):
        attribution, caption, figure_flag, headings, image_url, label, name, referee_id, section_url = get_graph_metadata(graph)
        text_block = text_block.strip()
     
        if text_block.startswith(name):
            chunk = to_chunk(text_block, section_url, referee_id, headings)
            docs.append(chunk)
    
    with open("../data/processed/graphs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=4)

if __name__ == "__main__":
    main()