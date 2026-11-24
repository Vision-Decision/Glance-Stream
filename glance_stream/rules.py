"""Standing rules over a video stream.

A rule is asked as one yes/no question ("is the rule being violated right now?") at every readout; the model answers
with p(yes) from the first answer token. Cross-time rules ("after A, B is not allowed", "A must be followed by B within
T s", ...) are answered end to end from the stream state: the model was trained on these exact rule types, and there is
no separate event detector or state machine at inference time.

Rule types (the wording the model was trained and evaluated with is `Rule.template()`):

    now              {X}        X is happening right now
    after_then       {A, B}     B happens after A has happened at least once
    b_without_a      {A, B, T}  B starts with no A in the T seconds before it
    a_not_followed   {A, B, T}  A happened and no B followed within T seconds (violated from the deadline until B)
    count_in_window  {X, N, T}  X started at least N times within the last T seconds
    while_state      {A, C, B}  B happens between an A and the next C
    custom           {text}     your own yes/no question or rule sentence, asked as is

`Trigger` turns the per-readout p(yes) into alerts (k consecutive readouts at or above the threshold, then a cooldown),
the same rule the evaluation used. `violations` is the reference rule engine that computes ground-truth violation
intervals from timestamped events; use it to score the model on your own annotated videos.
"""
from dataclasses import dataclass, field

TYPES = ("now", "after_then", "b_without_a", "a_not_followed", "count_in_window", "while_state", "custom")
FIELDS = dict(now="X", after_then="AB", b_without_a="ABT", a_not_followed="ABT", count_in_window="XNT",
              while_state="ACB", custom="")


@dataclass
class Rule:
    type: str
    X: str = None
    A: str = None
    B: str = None
    C: str = None
    T: float = None
    N: int = None
    text: str = None          # custom wording (any rule type); asked instead of the template
    name: str = None          # label in outputs; defaults to the question
    box: str = None           # describe a person to localise while this rule is on (see README: boxes)
    threshold: float = 0.5    # on p(yes)
    k: int = 2                # consecutive readouts at or above the threshold before an alert
    cooldown_s: float = 5.0   # no second alert for this rule within this many seconds

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"unknown rule type {self.type!r}; expected one of {TYPES}")
        if self.type == "custom" and not self.text:
            raise ValueError("a custom rule needs text")
        missing = [f for f in FIELDS[self.type] if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.type} rule needs {', '.join(missing)}")
        if self.name is None:
            self.name = self.text or self.template()

    # -- constructors ------------------------------------------------------------------------------------------------
    @classmethod
    def now(cls, x, **kw):
        """X is happening right now, e.g. Rule.now("someone is cutting vegetables")."""
        return cls("now", X=x, **kw)

    @classmethod
    def after(cls, a, forbidden, **kw):
        """after A has happened, B is not allowed."""
        return cls("after_then", A=a, B=forbidden, **kw)

    @classmethod
    def requires_before(cls, b, a, within, **kw):
        """B is only allowed if A happened in the `within` seconds before it."""
        return cls("b_without_a", A=a, B=b, T=within, **kw)

    @classmethod
    def must_follow(cls, a, b, within, **kw):
        """whenever A happens, B must follow within `within` seconds."""
        return cls("a_not_followed", A=a, B=b, T=within, **kw)

    @classmethod
    def at_most(cls, x, times, within, **kw):
        """X may happen at most `times` times in any `within` seconds (violated at the times+1-th start)."""
        return cls("count_in_window", X=x, N=times + 1, T=within, **kw)

    @classmethod
    def forbidden_between(cls, a, c, b, **kw):
        """between A and the next C, B is not allowed."""
        return cls("while_state", A=a, C=c, B=b, **kw)

    @classmethod
    def custom(cls, text, **kw):
        """your own yes/no question, e.g. "Is the stove on with nobody in the kitchen?" (untested wording)."""
        return cls("custom", text=text, **kw)

    # -- wording -----------------------------------------------------------------------------------------------------
    def template(self):
        t = self.type
        T = _num(self.T)
        if t == "now":
            return f"Is this happening right now: {self.X}?"
        if t == "after_then":
            return f"Rule: after '{self.A}' has happened, '{self.B}' is not allowed. Is the rule being violated right now?"
        if t == "b_without_a":
            return (f"Rule: '{self.B}' is only allowed if '{self.A}' happened in the {T} seconds before it. "
                    "Is the rule being violated right now?")
        if t == "a_not_followed":
            return (f"Rule: whenever '{self.A}' happens, '{self.B}' must follow within {T} seconds. "
                    "Is the rule being violated right now?")
        if t == "count_in_window":
            return (f"Rule: '{self.X}' may happen at most {self.N - 1} times in any {T} seconds. "
                    "Is the rule being violated right now?")
        if t == "while_state":
            return (f"Rule: between '{self.A}' and the next '{self.C}', '{self.B}' is not allowed. "
                    "Is the rule being violated right now?")
        return self.text

    def question(self):
        return (self.text or self.template()).rstrip() + " Answer yes or no."

    def as_dict(self):
        return {k: v for k, v in vars(self).items() if v is not None}


def _num(x):
    return None if x is None else (int(x) if float(x).is_integer() else x)


@dataclass
class Trigger:
    """Alert rule used by the evaluation: k consecutive readouts with p >= threshold, then `cooldown_s` without alerts."""
    threshold: float = 0.5
    k: int = 2
    cooldown_s: float = 5.0
    run: int = 0
    last: float = field(default=-1e9)

    def update(self, t, p):
        self.run = self.run + 1 if p >= self.threshold else 0
        if self.run >= self.k and t - self.last >= self.cooldown_s:
            self.last, self.run = t, 0
            return True
        return False


# -- reference rule engine: ground truth from timestamped events ---------------------------------------------------------
MIN_LEN = 0.5  # violation intervals shorter than this are dropped (annotation jitter)


def _starts(ev, c):
    return sorted(s for k, s, e in ev if k == c)


def _ivs(ev, c):
    return sorted([s, e] for k, s, e in ev if k == c)


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def violations(rule, events, t_end):
    """rule: Rule or dict with "type" and its fields; events: [(name, start_s, end_s)] in stream time.
    -> sorted [[start, end]] intervals during which the rule counts as violated."""
    r = rule.as_dict() if isinstance(rule, Rule) else rule
    t, ev = r["type"], events
    if t == "now":
        out = _ivs(ev, r["X"])
    elif t == "after_then":
        a = _starts(ev, r["A"])
        out = [[max(s, a[0]), e] for s, e in _ivs(ev, r["B"]) if a and e > a[0]]
    elif t == "b_without_a":
        a = _starts(ev, r["A"])
        out = [[s, e] for s, e in _ivs(ev, r["B"]) if not any(s - r["T"] <= x <= s for x in a)]
    elif t == "a_not_followed":
        # violated from the deadline until B finally happens, A starts again, or the stream ends
        b, a, out = _starts(ev, r["B"]), _starts(ev, r["A"]), []
        for i, x in enumerate(a):
            if any(x < y <= x + r["T"] for y in b) or x + r["T"] >= t_end:
                continue
            nb = next((y for y in b if y > x + r["T"]), t_end)
            na = a[i + 1] if i + 1 < len(a) else t_end
            out.append([x + r["T"], min(nb, na, t_end)])
    elif t == "count_in_window":
        st, n, T = _starts(ev, r["X"]), r["N"], r["T"]
        out = [[st[i + n - 1], st[i] + T] for i in range(len(st) - n + 1) if st[i + n - 1] - st[i] <= T]
    elif t == "while_state":
        on, off, states, i = sorted(_starts(ev, r["A"])), sorted(_starts(ev, r["C"])), [], 0
        while i < len(on):
            s = on[i]
            e = next((x for x in off if x > s), t_end)
            states.append([s, e])
            i = next((j for j in range(i, len(on)) if on[j] >= e), len(on))
        out = [[max(s, a), min(e, b)] for s, e in _ivs(ev, r["B"]) for a, b in states if min(e, b) > max(s, a)]
    else:
        raise ValueError(f"no reference semantics for rule type {t!r}")
    return [iv for iv in merge([[max(0.0, s), min(t_end, e)] for s, e in out]) if iv[1] - iv[0] >= MIN_LEN]
