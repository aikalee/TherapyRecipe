from bs4 import BeautifulSoup

import sys
import re
import os
import json

chunk_id = 1

filename = 'initial_cleanedup_guideline_without_tabs.html'
# if there's no input from command line, use the default filename
if len(sys.argv) > 1:
    filename = sys.argv[1]
    
with open(filename, "r", encoding="utf-8") as f:
    html = f.read()


# # I manually substitute the Level 1, Level 2, Level 3, Level 4 link with text like (Leve 1), (Level 2), (Level 3), (Level 4)
# level1 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/62befe587468/10.1177_07067437241245384-img1.jpg'
# level2 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/b9ea5ad77490/10.1177_07067437241245384-img2.jpg'
# level3 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/5be38aafe33f/10.1177_07067437241245384-img3.jpg'
# level4 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/68e56cd87632/10.1177_07067437241245384-img4.jpg'


soup = BeautifulSoup(html, "html.parser")
output = []


# parse the h1 title
title = soup.find("h1")
if title:
    title = title.decode_contents().replace('\n', '')
    
output.append({
    "text": title,
    "metadata": {
        "section": "title",
        "type": "title",
        "chunk_index": chunk_id,
        "headings": "Title",
        "referenced_tables": '',
  }
})
chunk_id += 1


# parse the main body
for p in soup.find_all("p"):
    parent_sec = p.find_parent("section", id=True)
    sec_id = parent_sec.get("id") if parent_sec else None
    
    #TODO: I'm currently ignoring the HTML tables, becuase AIKA's processing them.
    #get the headings
    
    # get the closest heading
    heading = p.find_previous_sibling(lambda tag: bool(re.match(r'^h[2-6]$', tag.name)))
    headings = heading.get_text(strip=True) if heading else ''
    
    #while parent still has parents
    while parent_sec:
        # print(f"parent_sec: {parent_sec.get('id')}")
        heading = parent_sec.find_previous_sibling(lambda tag: bool(re.match(r'^h[2-6]$', tag.name)))
        if heading:
            headings = heading.get_text(strip=True) + ' > ' + headings
        parent_sec = parent_sec.find_parent("section", id=True)
    headings = headings.strip()
    # print(f"headings: {headings}")
    
    #get the text with <href> links and other tags
    # text = p.get_text(strip=True)
    text = p.decode_contents().replace('\n', ' ')
    # print(f"text: {text}\n\n")
    
    # mark Type of the text
    type = 'paragraph'
    
    if 'table' in sec_id:  # either table image or table in HTML format
        if p.get('class') and 'img-box' in p.get('class'):
            type = 'table image '
        else: 
            continue
        text = str(p)

        # I only have 2 here with parent = <figure>, most of img's parent are <section id = 'table...'>
    elif p.get('class') and 'img-box' in p.get('class'):
        type = 'figure image'
        text = str(p)
        sec_id = p.find_parent("figure", id=True).get("id")
    
    chunk = {
        "text": "From section: "+ headings.replace('\n', ' ') + " > paragraph id: " + str(chunk_id) + "\n"+ text,
        "metadata": {
            "section": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11351064/#" + sec_id,
            "type": type,
            "chunk_index": chunk_id,
            "headings": headings.replace('\n', ' '),
            "referenced_tables": '',
        }
    }
    output.append(chunk)
    chunk_id += 1
    
    # for i in current_chunk:
    #     print(f"{i}: {current_chunk[i]}")
    # print("\n\n")
    # dump output to json
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"output.json file created with {len(output)} chunks.")