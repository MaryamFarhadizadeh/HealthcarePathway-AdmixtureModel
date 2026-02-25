from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Node:
    parents: list["Node"]
    children: list["Node"] 
    states: dict[str, int]
    id: str
    count: int = 0
    patid: list[str] = field(default_factory=list)

    nextid = 1
    startstate = "['Start']"

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.states.keys == other.states.keys
        return False

    @classmethod
    def createStartNode(cls):
        startnode = Node([], [], {Node.startstate: 0}, "ID" + str(Node.nextid), 0, [])
        Node.nextid += 1
        return startnode

    def addChild(self, states):
        new_node = Node([self], [], {str(states): 0}, "ID" + str(Node.nextid), 0, [])
        Node.nextid += 1
        self.children.append(new_node)
        return new_node

    def addPatient(self, pat: "Patient", pos: int = 0):
        if pos == 0:
            self.count += 1
            self.states[Node.startstate] += 1
            nextindex = pat.nextTimeIndex(pos)
            pos = nextindex
        if pos < len(pat.states):
            nextindex = pat.nextTimeIndex(pos)
            curstates = str(sorted(pat.states[pos:nextindex]))
            foundchild = None
            for child in self.children:
                if curstates in child.states.keys():
                    foundchild = child
                    foundchild.patid.append(pat.id)
                    break

            if not foundchild:
                foundchild = self.addChild(curstates)
                foundchild.patid.append(pat.id)

            foundchild.count += 1
            foundchild.states[curstates] += 1
            foundchild.addPatient(pat, nextindex)

    def addImportances(self, importances):
        if len(self.children) > 0:
            childn = 0
            for child in self.children:
                childn += child.count
            for child in self.children:
                for key in child.states.keys():
                    curfrac = child.states[key] / childn
                    curweightedn = (0.5 - abs(0.5 - curfrac)) * childn
                    if key in importances.keys():
                        importances[key] += curweightedn
                    else:
                        importances[key] = curweightedn
                child.addImportances(importances)

    def getImportances(self):
        importances = {}
        self.addImportances(importances)
        return importances

    def mergeForward(self, othernode: "Node"):
        self.count += othernode.count
        for otherstate in othernode.states.keys():
            if otherstate in self.states.keys():
                self.states[otherstate] += othernode.states[otherstate]
            else:
                self.states[otherstate] = othernode.states[otherstate]

        self.patid = list(set(self.patid) | set(othernode.patid))

        for otherchild in othernode.children:
            otherstates = sorted(otherchild.states.keys())
            foundpartner = False
            for mychild in self.children:
                if sorted(mychild.states.keys()) == otherstates:
                    mychild.mergeForward(otherchild)
                    foundpartner = True
            if not foundpartner:
                otherchild.parents = [self]
                self.children.append(otherchild)

    def mergeBackward(self, othernode: "Node", protected):
        self.count += othernode.count
        for otherstate in othernode.states.keys():
            if otherstate in self.states.keys():
                self.states[otherstate] += othernode.states[otherstate]
            else:
                self.states[otherstate] = othernode.states[otherstate]

        self.patid = list(set(self.patid) | set(othernode.patid))

        for otherparent in othernode.parents:
            foundparent = False
            for parent in self.parents:
                if parent.id == otherparent.id:
                    foundparent = True
                    break

            if not foundparent:
                self.parents.append(otherparent)
                if self not in otherparent.children:
                    otherparent.children.append(self)

            for i in reversed(range(len(otherparent.children))):
                if otherparent.children[i] == othernode:
                    otherparent.children.pop(i)
                    break

        if len(self.parents) > 1:
            for i in reversed(range(1, len(self.parents))):
                for j in range(i):
                    if self.parents[i].states.keys() == self.parents[j].states.keys():
                        self.parents[j].mergeBackward(self.parents[i], protected)
                        self.parents.pop(i)
                        break

    def statesMatch(self, matchstates):
        return len(set(self.states.keys()) & set(matchstates)) > 0

    def smallestState(self):
        smstate = ""
        smlen = 0
        mykeys = list(self.states.keys())
        for i in range(len(mykeys)):
            curlen = len(eval(mykeys[i]))
            if curlen > smlen:
                smstate = mykeys[i]
                smlen = curlen
        return smstate

    @classmethod
    def collapseNodes(cls, nodes, protected: list[str], forward=True):
        protectedindex = []
        collapseindex = []
        popindex = []
        for i in range(len(nodes)):
            if nodes[i].statesMatch(protected):
                notadded = True
                cursmstate = nodes[i].smallestState()
                for j in range(i + 1, len(nodes)):
                    if nodes[j].smallestState() == cursmstate:
                        collapseindex.append(i)
                        notadded = False
                        break
                if notadded:
                    protectedindex.append(i)
            else:
                collapseindex.append(i)

        if len(collapseindex) > 0:
            for i in collapseindex:
                bestfit = -1
                bestoverlap = 0
                for j in protectedindex:
                    pkeys = nodes[i].states.keys()
                    ckeys = nodes[j].states.keys()
                    for pkey in pkeys:
                        for ckey in ckeys:
                            curoverlap = len(set(eval(pkey)) & set(eval(ckey)))
                            if curoverlap > bestoverlap:
                                bestfit = j
                                bestoverlap = curoverlap

                if bestfit == -1 and not forward:
                    for j in range(i + 1, len(nodes)):
                        if not nodes[j].statesMatch(protected):
                            bestfit = j

                if bestfit != -1:
                    if forward:
                        nodes[bestfit].mergeForward(nodes[i])
                    else:
                        nodes[bestfit].mergeBackward(nodes[i], protected)
                    popindex.append(i)

        if len(popindex) > 0:
            for i in sorted(popindex, reverse=True):
                nodes.pop(i)

        return nodes

    def popendnodes(self, nodes):
        for i in range(len(nodes)):
            if len(nodes[i].children) == 0 and nodes[i].states.keys() != '166':
                nodes.pop(i)

    def collapseVertically(self, protected: list[str]):
        self.children = Node.collapseNodes(self.children, protected)
        for child in self.children:
            child.collapseVertically(protected)

    def collapseEnds(self, protected: list[str]):
        endnodes = []
        for node in self.collectNodes([]):
            if len(node.children) == 0:
                endnodes.append(node)
        Node.collapseNodes(endnodes, protected, forward=False)

    def collapseHorizontally(self, protected):
        if not (len(self.parents) == 0 or self.statesMatch(protected)):
            i = 0
            while i < len(self.children):
                nextchild = self.children[i]
                if not nextchild.statesMatch(protected):
                    for child in nextchild.children:
                        self.children.append(child)
                    for state in nextchild.states.keys():
                        if state in self.states.keys():
                            self.states[state] += nextchild.states[state]
                        else:
                            self.states[state] = nextchild.states[state]
                    self.children.pop(i)
                else:
                    i += 1
            self.collapseVertically(protected)
        for child in self.children:
            child.collapseHorizontally(protected)

    def prune(self, relthresh, absthresh):
        if len(self.children) > 0:
            childn = 0
            for child in self.children:
                childn += child.count
        for i in reversed(range(len(self.children))):
            if self.children[i].count < absthresh or self.children[i].count / childn < relthresh:
                self.children.pop(i)
        for child in self.children:
            child.prune(relthresh, absthresh)

    def pruneNonDischargeNodes(self, discharge_state='[166]'):
        for node in self.collectNodes([]):
            if len(node.children) == 0 and discharge_state not in node.states.keys():
                for parent in node.parents:
                    if node in parent.children:
                        parent.children.remove(node)

    def collectNodes(self, nodelist: list["Node"] = []):
        if not self.id in [node.id for node in nodelist]:
            nodelist.append(self)
            for child in self.children:
                child.collectNodes(nodelist)
        return nodelist

@dataclass
class Patient:
    states: list[int]
    tpoints: list[int]
    id: str

    def nextTimeIndex(self, startindex: int = 0):
        nextindex = startindex + 1
        for curindex in range(startindex + 1, len(self.tpoints)):
            if self.tpoints[curindex] != self.tpoints[curindex - 1]:
                break
            else:
                nextindex += 1
        return nextindex

# -----------------------
# Helpers for plotting
# -----------------------
def listToStr(li):
    res = ""
    for i in range(len(li)):
        res += str(li[i])
        if i < len(li) - 1:
            res += ", "
    return res
