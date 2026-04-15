"""
Compute the radial distribution function (RDF) between Li ions and anion
centers of mass using MDAnalysis InterRDF, producing the RDF data for Fig. 2.
"""
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import matplotlib.pyplot as plt
import sys
import os

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

anion_model = sys.argv[1]
T_list = [sys.argv[2]]
for T in T_list:
    system_dir = os.path.join(DATA_ROOT, anion_model, str(T))
    out = open(os.path.join(system_dir, "rdfP.txt"),'w')
    tpr = os.path.join(system_dir, "center.gro")
    xtc = os.path.join(system_dir, "center.xtc")
    u = mda.Universe(tpr, xtc)
    #u.atoms[u.atoms.types == 'LI'].masses = 1234
    ref = u.select_atoms("name Li")
    sel = u.select_atoms("name P")
    rdf = InterRDF(ref, sel, nbins=300)
    rdf.run()
    num_bins = len(rdf.results.bins)
    for i in range(num_bins):
        out.write(f"{rdf.results.bins[i]:.3f}   {rdf.results.rdf[i]:.3f}\n")
    out.close()
