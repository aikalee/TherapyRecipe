## How we processed data

1. save html page
2. delete head and tail of the file. Start from title and delete the sections after ## conclusion(from supplementary materials). Delete the french abstract
3. Observe and identify the link, and h1/h2/h3 tag to identify the hierarchy of the data
  - get headings, concatnate all the headings together and pre-pend it to the actual paragraph texts.
4. Analyze special element in the texts: table image and HTML Table:
    - table image
        - table embedded in the page is accessed from link.
        - So far, we grab the link and send it to Vision model. I'll say send it to Chat gpt. convert the table to CSV/markdown. and have gpt generate notes to suggest that: this is a csv/markdown table
          - we observe a lot of green circles pictures that represents level1, level2 and 3,4
          - These pics are saved as link in the the html. We search for the links and then replace them with text
    - html table
        - send it to chat gpt also? parse it to csv/markdown format so it's easier for llm to understand

    - I also manually merge caption under the figure
    - some paragraphs in the Box is separated in different chunks. I think it would be better to merge them into one chunk. The longest of them is arond 250 words(around 350 tokens)

5. Abbreviation substitution. Detect all the abbreviation in the text and replace them with full definition.

# As we explore the data, design schema of the json structure database

{
  "text": "The results are summarized in Table 2...",
  "embedding": [...],
  "metadata": {
    "section": "Results",
    "type": "paragraph",
    "chunk_index": 42,
    "headings": "Abstract/Background",
    "referenced_tables": ["table_2"]
  }
}