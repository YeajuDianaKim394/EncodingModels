from praatio import textgrid

def records2tg(records: list): #, xmin: float = 0.0, xmax: float = 1e3):
    """General purpose function to convert dictionary records into TextGrid.

    Makes TextGrid min and max timestamp the same for all tiers.

    records = [
        {
            'speaker': 1,
            'onset': 0.5,     # in seconds
            'offset': 3.0,      # in sectonds
            'text':, 'hello world'
        },
        ...
    ]
    
    See:
    - https://nbviewer.org/github/timmahrt/praatIO/blob/main/tutorials/tutorial1_intro_to_praatio.ipynb
    - https://pypi.org/project/praatio/
    - https://github.com/zkokaja/b2b/blob/main/code/transcribe.py
    """
    xmin = records[0]['onset']
    xmax = records[-1]['offset']
    tg = textgrid.Textgrid(xmin, xmax)
    speakers = set(r['speaker'] for r in records)
    for speaker_id in speakers:
        entries = []
        for record in records:
            if record['speaker'] == speaker_id:
                entries.append([record['onset'], record['offset'], record['text']])
        tier = textgrid.IntervalTier(str(speaker_id), entries) #, minT=xmin, maxT=xmax)
        tg.addTier(tier)
    return tg
