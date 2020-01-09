# Deterministic Finite Automaton (DFA)
ERROR = "ERROR"
Q = ["start", "1-0", "0-1", "deuce", "win", "lose"]
alphabet = ["win", "lose"]

def transition(q, a):
    if not q in Q or not a in alphabet:
        return ERROR

    if q == "start":
        if a == "win":
            return Q[1]
        else:
            return Q[2]
    if q == "1-0":
        if a == "win":
            return Q[4]
        else:
            return Q[3]
    elif q == "0-1":
        if a == "win":
            return Q[3]
        else:
            return Q[5]
    elif q == "deuce":
        if a == "win":
            return Q[1]
        else:
            return Q[2]
    else:
        return ERROR

input_string = ["win", "lose", "lose", "win", "win", "win"]

def extended_transition(input_string):
    q = Q[0]
    if len(input_string) != 0:
        for s in input_string:
            q = transition(q, s)

    return q

print (extended_transition(input_string))