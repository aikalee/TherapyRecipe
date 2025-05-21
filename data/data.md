## How we processed data

1. save html page
2. delete head and tail of the file. Start from title and delete the sections after ## conclusion(from supplementary materials). Delete the french abstract
3. Observe and identify the link, and h1/h2/h3 tag to identify the hierarchy of the data
4. Analyze special element in the texts: table image and HTML Table:
    - table image
        - table embedded in the page is accessed from link.
        - grab the link and send it to OCR? I'll say send it to Chat gpt. convert the table to CSV/markdown. and have gpt generate notes to suggest that: this is a csv/markdown table
        - 我们同时观察到很多表示level-1，level-2的图片。
        - 首先，可以对于inline level图片进行替换，identify 不同level图片的
        - parse相关图片的时候，prompt喂给chat-gpt让它在识别到图片的时候，用文字level1,level2,level3表示。
    - html table
        - send it to chat gpt also? parse it to csv/markdown format so it's easier for llm to understand

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


{
        "section_id": "section2-07067437241245384", # 距离该<p><\p>最近的section id

        "metadata": "Abstract/Background"
        "Text": CANMAT convened a guidelines editorial group, comprised of
            academic clinicians (<em>n</em> = 43, representing diversity in
            seniority, region, and expertise) and a patient partner (KS), to
            direct and manage the 2023 update. Methods, search strategies,
            and evidence tables are detailed on the project website on the
            Open Science Framework (OSF) (<a href="https://osf.io/8tfkp/" class="usa-link usa-link--external"
                data-ga-action="click_feat_suppl" target="_blank" rel="noopener noreferrer">https://osf.io/8tfkp/</a>).
            High-quality, large-sample randomized controlled trials (RCTs)
            remain the gold standard for evidence. However, owing to the
            sheer volume of available RCTs, we prioritized systematic
            reviews and meta-analyses that synthesize many RCTs, for much of
            the evidence used and cited in these guidelines. We also
            recognize the limitations of meta-analyses; hence, we
            complemented them with results from large RCTs when making
            recommendations.
}