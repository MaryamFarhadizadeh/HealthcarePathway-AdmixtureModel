import numpy as np
import pandas as pd
import copy
import math
import graphviz
from pathlib import Path
from .tree import Node, Patient, listToStr

# ---------Define a color map for specific states---------
COLOR_MAP = {
    "0": "#8787BF",
    "166": "#5477A7",
    "106": "#B8C7D9",
    "106, 119": "#809FD1",
    "10, 119": "#D1ABCF",
    "30": "#8FC4D1",
    "54": "#8EEDC3",
    "91": "#4682B4",
    "164": "#9CF6FB",
}


#------------Loading data-----------------
def load_data(csv_path: str):
    csv_path = Path(csv_path)
    return pd.read_csv(csv_path)


#------------Train/Test split-----------------
def train_test_split(a_df, seed=35, train_ratio=0.8):
    rng = np.random.default_rng(seed)
    # Use a NumPy array to avoid pandas StringArray shuffle warnings.
    patient_ids = np.array(a_df['patient_num'].unique(), dtype=object)
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(train_ratio * n_total)

    train_ids = set(patient_ids[:n_train])
    test_ids  = set(patient_ids[n_train:])

    a_df_train = a_df[a_df['patient_num'].isin(train_ids)].copy()
    a_df_test  = a_df[a_df['patient_num'].isin(test_ids)].copy()

    print(f"Patients total: {n_total} | train: {len(train_ids)} | test: {len(test_ids)}")
    print(f"Rows total: {len(a_df)} | train rows: {len(a_df_train)} | test rows: {len(a_df_test)}")

    return a_df_train, a_df_test

# -----------Building Patient Objects-----------------
def build_patient_objects(a_df_subset):
    patvec = []
    for name, subdf in a_df_subset.groupby('patient_num'):
        pat = Patient(
            states=subdf['states'].astype(int).tolist(),
            tpoints=subdf['pathOrder'].astype(int).tolist(),
            id=str(name)
        )
        patvec.append(pat)
    return patvec

#------------Building the prefix tree-----------------
def build_prefix_tree(patvec_train):
    Node.nextid = 1
    mystart = Node.createStartNode()

    for patient in patvec_train:
        mystart.patid.append(patient.id)

    for patient in patvec_train:
        mystart.addPatient(patient)

    return mystart


#------------Simplification Pipline-----------------
def run_simplification_pipeline(
    mystart,
    importance_keep_threshold=5.0,
    unimportant_threshold=10.0,
    prune_rel_threshold=0.1,
    prune_abs_threshold=5,
    manual_protected_state_ids=None,
):
    importances = mystart.getImportances()

    istates = [key for key in importances if importances[key] > importance_keep_threshold]
    unimportant_nodes = [key for key in importances if importances[key] <= unimportant_threshold]

    if manual_protected_state_ids:
        for state_id in manual_protected_state_ids:
            state_key = str([int(state_id)])
            if state_key not in istates:
                istates.append(state_key)

    print("Unimportant nodes (train):", unimportant_nodes)
    print("Nodes before collapse/prune:", len(mystart.collectNodes([])))

    mystart.collapseVertically(istates)
    print("After collapseVertically:", len(mystart.collectNodes([])))

    mystart.collapseHorizontally(istates)
    print("After collapseHorizontally:", len(mystart.collectNodes([])))

    mystart.prune(prune_rel_threshold, prune_abs_threshold)
    print("After prune:", len(mystart.collectNodes([])))

    beforeends = copy.deepcopy(mystart)
    mystart.collapseEnds(istates)
    print("After collapseEnds:", len(mystart.collectNodes([])))

    mystart.pruneNonDischargeNodes()
    print("After pruneNonDischargeNodes:", len(mystart.collectNodes([])))

    return mystart, istates

#------------Rendering the graph-----------------
#------------Rendering the graph-----------------
def render_train_graph(mystart, istates, show_legend=True):

    protected = istates.copy()
    protected.append(Node.startstate)

    allnodes = mystart.collectNodes([])

    node_attributes = {
        'shape': 'rectangle',
        'style': 'filled',
        'fontname': 'Helvetica'
    }

    dot = graphviz.Digraph(comment='Typical Pathways (TRAIN)')
    dot.graph_attr["rankdir"] = "LR"

    # -------- ADD NODES --------
    for node in allnodes:

        # Root node
        if len(node.parents) == 0:
            label = f"0: {node.count}"
            color = COLOR_MAP["0"]

        else:
            pmatch = list(set(node.states.keys()) & set(protected))

            if len(pmatch) > 0:
                state_str = listToStr(eval(pmatch[0]))
                color = COLOR_MAP.get(state_str, "lightblue")

                label = f"{state_str}: {sum(node.states.values())}"

                if len(node.states) > 1:
                    label += ";"

                for key in sorted(node.states.keys()):
                    if key != pmatch[0]:
                        diff = list(set(eval(key)) ^ set(eval(pmatch[0])))
                        label += f"\n{listToStr(diff)}: {node.states[key]}"

            elif len(node.states) == 1:
                key = list(node.states.keys())[0]
                label = f"{listToStr(eval(key))}: {node.states[key]}"
                color = "lightblue"

            else:
                label = "B:" + str(node.count) + "\n" + "\n".join(
                    f"{listToStr(eval(key))}: {node.states[key]}"
                    for key in sorted(node.states.keys())
                )
                color = "lightblue"

        dot.node(node.id, label, **node_attributes, fillcolor=color)

    # -------- LEGEND --------
    if show_legend:
        legend_label = f"""<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
          <TR><TD COLSPAN="2"><B>Legend</B></TD></TR>
          <TR><TD BGCOLOR="{COLOR_MAP["0"]}"> </TD><TD>Prostate cancer diagnosis</TD></TR>
          <TR><TD BGCOLOR="{COLOR_MAP["10, 119"]}"> </TD><TD>Fusion biopsy</TD></TR>
          <TR><TD BGCOLOR="{COLOR_MAP["30"]}"> </TD><TD>Cystography</TD></TR>
          <TR><TD BGCOLOR="{COLOR_MAP["54"]}"> </TD><TD>PET-CT</TD></TR>
          <TR><TD BGCOLOR="{COLOR_MAP["106"]}"> </TD><TD>Open Prostatectomy</TD></TR>
          <TR><TD BGCOLOR="{COLOR_MAP["106, 119"]}"> </TD><TD>Robotic-Assisted Prostatectomy</TD></TR>
          <TR><TD BGCOLOR="{COLOR_MAP["164"]}"> </TD><TD>Multimodal psychotherapeutic treatment</TD></TR>
          <TR><TD BGCOLOR="{COLOR_MAP["166"]}"> </TD><TD>Hospital discharge</TD></TR>
        </TABLE>>"""

        dot.attr(label=legend_label, labelloc="t", labeljust="r")

    # -------- ADD EDGES --------
    # Allocate incoming edge counts per child so the sum of incoming
    # edge labels equals child.count (and therefore node label count).
    edge_counts = {}
    allnode_ids = {n.id for n in allnodes}
    for child in allnodes:
        if len(child.parents) == 0:
            continue

        # Use only active links that are actually drawable in this graph.
        active_parents = [
            p for p in child.parents
            if p.id in allnode_ids and child in p.children
        ]
        # Fallback: derive parents from children pointers if parent list is stale.
        if len(active_parents) == 0:
            active_parents = [p for p in allnodes if child in p.children]

        if len(active_parents) == 0:
            continue

        overlaps = []
        for parent in active_parents:
            overlap = len(set(parent.patid) & set(child.patid))
            overlaps.append((parent, overlap))

        total_overlap = sum(v for _, v in overlaps)
        target = int(child.count)

        if total_overlap <= 0:
            # Fallback: split evenly if overlaps are unavailable.
            base = target // len(active_parents)
            remainder = target - (base * len(active_parents))
            for i, (parent, _) in enumerate(overlaps):
                edge_counts[(parent.id, child.id)] = base + (1 if i < remainder else 0)
            continue

        # Proportional allocation with deterministic rounding to exactly target.
        raw_alloc = []
        for parent, overlap in overlaps:
            share = (target * overlap) / total_overlap
            floor_share = int(math.floor(share))
            frac = share - floor_share
            raw_alloc.append([parent, floor_share, frac])

        assigned = sum(x[1] for x in raw_alloc)
        remainder = target - assigned
        raw_alloc.sort(key=lambda x: x[2], reverse=True)
        for i in range(max(0, remainder)):
            raw_alloc[i % len(raw_alloc)][1] += 1

        for parent, count_alloc, _ in raw_alloc:
            edge_counts[(parent.id, child.id)] = count_alloc

    for node in allnodes:
        for child in node.children:
            ecount = edge_counts.get((node.id, child.id), 0)
            if ecount <= 0:
                continue

            b = max(0, (ecount - 5))
            penwidth = math.ceil((1 + math.sqrt(1 + (b * 10))) * 0.15)

            dot.edge(
                node.id,
                child.id,
                label=str(ecount),
                penwidth=str(penwidth),
                color="grey",
                fontname="Helvetica"
            )

    return dot
