import numpy as np 
import scipy.sparse as sp 
import sqlite3 as sql
from ..regex_parse import extract_abstracts

'''
    data pipeline script to construct a graph relationship between citations based on PMIDs
'''

def load_pmids(path):
    '''
    load list of pmids into np.int64 
    instantiate dict for constructing CSR
    return n-array(len=($"number of pmids"), dtype=np.int64), dict( PMID: index )
    '''
    pmids = np.loadtxt(path, dtype=np.int64)
    pmid_idx = {int(p): i for i, p in enumerate(pmids)}
    return pmids, pmid_idx

def load_abstracts(path, pmid_idx, keep_labels=False):
    abstracts_dict = extract_abstracts(path, keep_labels=keep_labels)
    abstracts = np.empty(len(pmid_idx), dtype=object)
    for pmid, text in abstracts_dict.items():
        if pmid in pmid_idx:
            abstracts[pmid_idx[pmid]] = text
    return abstracts

def fetch_citations(db_path, pmid_idx):
    '''
    reads in list of citations from pmid
    use a temp table to fetch cited_by from icite.db 
    return [($cited_pmid, "$citing_pmid $citing_pmid ... "), ...]
    '''
    src = pmid_idx.keys()
    pmids_rows = zip(src)
    with sql.connect(db_path) as db:
        curs = db.cursor()
        curs.execute("CREATE TEMP TABLE temp_pmids(pmid INTEGER PRIMARY KEY)")
        curs.executemany("INSERT INTO temp_pmids VALUES(?)", pmids_rows)
        curs.execute('''SELECT p.pmid, p.cited_by
                        FROM papers p
                        JOIN temp_pmids t ON p.pmid = t.pmid
                        WHERE p.cited_by IS NOT NULL AND p.cited_by != ""''')
        # pmid_tbl = [($cited_pmid, "$citing_pmid $citing_pmid")]
        pmid_tbl = curs.fetchall()
        curs.close()
    return pmid_tbl

def fetch_years(db_path, pmid_idx):
    '''
    reads in publication year (iCite only stores year-level granularity, no
    month/day) for every pmid in pmid_idx
    return {pmid: year}, omitting pmids with a NULL year in iCite
    '''
    src = pmid_idx.keys()
    pmids_rows = zip(src)
    with sql.connect(db_path) as db:
        curs = db.cursor()
        curs.execute("CREATE TEMP TABLE temp_pmids(pmid INTEGER PRIMARY KEY)")
        curs.executemany("INSERT INTO temp_pmids VALUES(?)", pmids_rows)
        curs.execute('''SELECT p.pmid, p.year
                        FROM papers p
                        JOIN temp_pmids t ON p.pmid = t.pmid
                        WHERE p.year IS NOT NULL''')
        rows = curs.fetchall()
        curs.close()
    return {int(pmid): int(year) for pmid, year in rows}

def build_edges(pmid_tbl, pmid_idx, weight_fn=None):
    '''
    build a list of all edges in db
    returns [(cited_idx, citing_idx, weight), ...]
    weight_fn(cited_pmid, citing_pmid) -> float, defaults to 1.0
    '''
    edges = []
    for row in pmid_tbl:
        cited_pmid = int(row[0])
        citers = [int(i) for i in row[1].split(" ") if int(i) in pmid_idx]
        for citing_pmid in citers:
            w = weight_fn(cited_pmid, citing_pmid) if weight_fn else 1.0
            edges.append((pmid_idx[cited_pmid], pmid_idx[citing_pmid], w))
    return edges

def build_csr(edges, n):
    if not edges:
        raise ValueError("No edges found — check that PMIDs overlap with iCite cited_by data")
    rows, cols, data = zip(*edges)
    return sp.csr_matrix(
        (np.array(data, dtype=np.float32), (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n)
    )
