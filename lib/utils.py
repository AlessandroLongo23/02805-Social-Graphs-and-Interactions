import sys
sys.path.append("..")

import lib.regexPatterns as rgx
import networkx as nx
import urllib.parse
import json
import re
from tqdm import tqdm
import urllib.request
import os

def getPerformerGraph():
    performers = getPerformers()
    calculateLinks(performers)

    G = nx.DiGraph()

    for performer in performers:
        performer_original = performer[0]
        G.add_node(performer_original, length_of_content=performer[2])
        performer = urllib.parse.quote(performer_original.encode('utf-8'))
        performer = performer.replace('/', '%2F')

        with open(f'../data/rock/performers/{performer}.json', 'r', encoding='utf-8') as f:
            html_content = json.load(f)
            html_content = html_content["parse"]["text"]["*"]

        links = re.findall(rgx.hrefPattern, html_content)

        replace_dict = {
            '%2526': '&',
            '%2527': "'",
            '%28': '(',
            '%29': ')',
            '_': ' ',
            '%2C': ',',
            '%21': '!',
            '%23': '#',
            '%25C3%2596': 'Ö',
            '%25C3%25B6': 'ö',
            '%25C3%259C': 'Ü',
            '%25C3%25BC': 'ü',
            '%25C3%25BF': 'ÿ',
        }

        for link in links:
            link = link[6:]
            link = urllib.parse.quote(link.encode('utf-8'))
            for key, value in replace_dict.items():
                link = link.replace(key, value)

            if '%26' in link:
                continue
            link = link.replace('%25E2%2580%2593', '–')

            for other in performers:
                if link == other[0]:
                    G.add_edge(performer_original, other[0])

    return G


def getPerformers():
    try:
        with open('../data/rock/performers_data.json', 'r', encoding='utf-8') as f:
            performers = json.load(f)
    except:
        with open('../data/rock/performers.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        performers = re.findall(rgx.extractPerformersPattern, text)
        for i in range(len(performers)):
            performers[i] = list(performers[i])

    return performers

def calculateLinks(performers):
    for p in tqdm(performers):
        performer = p[0]
        performer = urllib.parse.quote(performer.encode('utf-8'))
        performer = performer.replace('/', '%2F')

        file_path = f'../data/rock/performers/{performer}.json'

        if os.path.exists(file_path):
            continue

        baseurl = "https://en.wikipedia.org/w/api.php?"
        params = {
            "action": "parse",
            "page": performer,
            "format": "json",
            "prop": "text",
            "disableeditsection": True,
            "disabletoc": True
        }

        query = baseurl + "&".join([f"{key}={value}" for key, value in params.items()])

        headers = {
            'User-Agent': '02805 Social Graphs and Interactions (longoa02@gmail.com) Python/3.x'
        }

        req = urllib.request.Request(query, headers=headers)
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode("utf-8"))

        html_content = data["parse"]["text"]["*"]

        content = re.sub(rgx.tagPattern, ' ', html_content)
        words = re.findall(rgx.wordPattern, content)
        p.append(len(words))

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    with open('../data/rock/performers_data.json', 'w', encoding='utf-8') as f:
        json.dump(performers, f, ensure_ascii=False, indent=4)


def extract_distribution(data):
    a, b = min(data), max(data)
    bins = [i for i in range(a, b + 1)]
    data = [d for d in data if d != 0]
    heights = [sum([1 if i == bin_ else 0 for i in data]) for bin_ in bins]
    return bins, heights
