from bs4 import BeautifulSoup
import json
import pandas as pd
import re


def get_table_metadata(table):

    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11351064/#"
    section = table.find_parent('section')

    section_id = section.find_parent('section').find_parent('section').get("id")
    section_url = url + section_id

    section_heading = section.find_parent('section').find_parent('section').select_one(f"[data-anchor-id={section_id}]").get_text()
    section_subheading = section.find_parent('section').select_one(".pmc_sec_title").get_text()
    headings = section_heading + " > " + section_subheading

    name = section.find("h4").get_text()
    caption = section.select_one('.caption p').get_text()
    label = name + " " + caption
    number = "_".join(re.findall(r"(.{1})\.", name)).lower()
    referee_id = f"table_{number}"


    footnotes = []

    # case 1: sup outside p
    for sup in section.select('.fn sup'):

        sibling = sup.find_next_sibling("p")

        if sibling:
            sup_text = sup.get_text(strip=True)
            fn_text = sibling.get_text()
            footnotes.append((sup_text, fn_text))
         
    

    # case 2: sup inside p
    for p in section.select('.fn p'):

        matches = re.findall(r"(?<=(\*|#))\s*(.*?)(?=\s\*|\s#|$)", p.get_text())

        if matches:
            footnotes.extend(matches)

    return name, caption, dict(footnotes), headings, label, referee_id, section_url


def get_table_data(table, footnotes):

    table_data = []

    rowspan_tracker = {}

    sups = footnotes.keys()

    subsec = ""

    for tr in table.find_all("tr"):

        row = []
        col_index = 0  

    
        while col_index in rowspan_tracker:
            value, remaining = rowspan_tracker[col_index]
            row.append(value)
            remaining -= 1
            if remaining == 0:
                del rowspan_tracker[col_index]
            else:
                rowspan_tracker[col_index] = (value, remaining)
            
            col_index += 1


        for cell in tr.find_all(["td", "th"]):


            cell_sups = []
            get_sups = cell.find_all("sup")
            cell_text = cell.get_text(separator="\n", strip=True)
            rowspan = int(cell.get("rowspan", 1))
            
            if "*" in cell_text:
                cell_sups.append("*")

            elif get_sups:
                cell_text = " ".join([t for t in cell_text.split("\n") if len(t) > 1])
                for sup in get_sups:
                    cell_sups.append(sup.get_text())                   

            if cell_sups:
                for sup in cell_sups:
                    cell_text += f" ({footnotes[sup]})"
            
                    
                    
            
            if int(cell.get("colspan", 1)) > 1:
                subsec = cell_text
                continue
           
            row.append(cell_text)

            if rowspan > 1:
                rowspan_tracker[col_index] = (cell_text, rowspan - 1)

            col_index += 1

    
        if row:
            if subsec:
                row.insert(0, subsec)
                
            table_data.append(row)

    return table_data


def to_text(table_data, label, caption):
  
    lines = []

    lines.append(f"[Table: {label}]")
    lines.append(f"Caption: {caption}")


    for i, row in enumerate(table_data[1:]):
        row_text = ", ".join([f"{k}: {v}" for k, v in zip(table_data[0], row)])
        lines.append(f"Row {i} — {row_text}")

    return "\n".join(lines)


def to_chunk(text_block, section_url, referee_id, headings):

    d = {
    "text": text_block,
    "metadata": {
        "section": section_url,
        "type": "figure table",
        "referee_id": referee_id,
        "headings": headings,
        }
    }
    return d


def tables_to_json(url="../data/raw/CANMAT_guidelines.html"):

    doc = []

    with open(url, encoding="utf-8") as f:
        html = f.read()
        soup = BeautifulSoup(html)
        tables = soup.find_all("table")
        table_count = len(tables)
        
    for tbl in tables:
        name, caption, footnotes, headings, label, referee_id, section_url, = get_table_metadata(tbl)
        table_data = get_table_data(tbl, footnotes)
        text_block = to_text(table_data, label, caption)
        chunk = to_chunk(text_block, section_url, referee_id, headings)
        doc.append(chunk)
    
    return doc


doc = tables_to_json()

with open("../data/processed/tables.json", "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=4)
