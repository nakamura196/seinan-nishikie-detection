"""外部の評価用に、NDL デジタルコレクションの錦絵の候補を集める（2026-11-02）。
NDL サーチの OpenSearch（dpid=ndl-dl）で「錦絵 <語>」を引き、題名・刊年・PID を data/external/ndl_candidates.tsv に書く。
使い方: python3 ndl_candidates.py"""
import urllib.parse, urllib.request, xml.etree.ElementTree as ET, csv, time
NS = {"dc": "http://purl.org/dc/elements/1.1/", "dcterms": "http://purl.org/dc/terms/", "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
      "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
seen = {}
for word in ("日清", "明治"):
    for start in range(1, 1000, 200):
        u = "https://ndlsearch.ndl.go.jp/api/opensearch?" + urllib.parse.urlencode({"dpid": "ndl-dl", "any": f"錦絵 {word}", "cnt": 200, "idx": start})
        items = ET.fromstring(urllib.request.urlopen(u).read()).findall(".//item")
        for it in items:
            sa = it.find("rdfs:seeAlso", NS)
            if sa is None: continue
            pid = sa.get(f"{{{NS['rdf']}}}resource").rsplit("/", 1)[1]
            seen.setdefault(pid, dict(pid=pid, query=word, title=it.findtext("dc:title", "", NS),
                                      creator=it.findtext("dc:creator", "", NS), issued=it.findtext("dcterms:issued", "", NS)))
        time.sleep(1)
        if len(items) < 200: break
with open("../data/external/ndl_candidates.tsv", "w") as f:
    w = csv.DictWriter(f, fieldnames=["pid", "query", "title", "creator", "issued"], delimiter="\t"); w.writeheader(); w.writerows(seen.values())
print(len(seen))
