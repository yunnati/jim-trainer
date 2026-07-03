import cv2, mediapipe as mp, numpy as np, math, time, threading, subprocess, sys, os
HAS_TTS = False
_tts_eng = None
def _init_voice():
    global HAS_TTS, _tts_eng
    try:
        import pyttsx3
        e = pyttsx3.init()
        e.setProperty('rate', 155)
        e.setProperty('volume', 1.0)
        _tts_eng = e
        HAS_TTS = True
        print("[Voice] pyttsx3 ready")
        return
    except Exception as ex:
        print(f"[Voice] pyttsx3 unavailable: {ex}")
    try:
        r = subprocess.run(["espeak","--version"], capture_output=True, timeout=2)
        if r.returncode == 0:
            _tts_eng = "espeak"; HAS_TTS = True
            print("[Voice] espeak ready"); return
    except Exception: pass
    try:
        r = subprocess.run(["say","test"], capture_output=True, timeout=2)
        if r.returncode == 0:
            _tts_eng = "say"; HAS_TTS = True
            print("[Voice] macOS say ready"); return
    except Exception: pass
    print("[Voice] No TTS — install pyttsx3 for voice")
_init_voice()
_v_busy = False
_v_last_txt = ""
_v_last_t   = 0.0
def _spk_worker(text):
    global _v_busy
    try:
        if _tts_eng == "espeak":
            subprocess.run(["espeak","-s","150",text], capture_output=True, timeout=8)
        elif _tts_eng == "say":
            subprocess.run(["say","-r","160",text], capture_output=True, timeout=8)
        elif _tts_eng:
            _tts_eng.say(text)
            _tts_eng.runAndWait()
    except Exception as e:
        print(f"[Voice] error: {e}")
    finally:
        _v_busy = False
def speak(text, gap=3.5):
    global _v_busy, _v_last_txt, _v_last_t
    if not HAS_TTS or _v_busy: return
    now = time.time()
    if text == _v_last_txt and now - _v_last_t < gap: return
    _v_last_txt = text; _v_last_t = now; _v_busy = True
    threading.Thread(target=_spk_worker, args=(text,), daemon=True).start()
LM = {
    "nose":0,"l_shoulder":11,"r_shoulder":12,
    "l_elbow":13,"r_elbow":14,"l_wrist":15,"r_wrist":16,
    "l_hip":23,"r_hip":24,"l_knee":25,"r_knee":26,
    "l_ankle":27,"r_ankle":28,
}
CONN = [
    ("l_shoulder","r_shoulder"),
    ("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
    ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
    ("l_shoulder","l_hip"),("r_shoulder","r_hip"),
    ("l_hip","r_hip"),
    ("l_hip","l_knee"),("l_knee","l_ankle"),
    ("r_hip","r_knee"),("r_knee","r_ankle"),
]
EXERCISES = {
    1: {
        "name":"Bicep Curl","muscle":"BICEPS","target_reps":12,
        "key_joints":["l_elbow","r_elbow"],
        # 3 landmark keys that form the rep angle
        "rep_j":("r_shoulder","r_elbow","r_wrist"),
        # "h2l" = starts extended (high angle), curls to low; "l2h" = opposite
        "rep_mode":"h2l","rep_start":145,"rep_end":65,
        "tips":["Elbows pinned to sides","Full extension at bottom","Squeeze at top","Wrists straight"],
        "injury":{"elbow_drift":"Elbow drift causes shoulder impingement. Keep elbows at your sides!"},
        "checks":[("Elbow Stability","elbow_drift",0.14),("Range of Motion","arm_rom",None),
                  ("Wrist Neutral","wrist_straight",0.09),("Shoulder Still","shoulder_stable",0.06)],
        "ref":[
            {"label":"START - Arms Extended","hl":["l_elbow","r_elbow"],"note":"~165 deg extended",
             "sk":{"head":(0.50,0.13),"l_shoulder":(0.37,0.29),"r_shoulder":(0.63,0.29),
                   "l_elbow":(0.33,0.47),"r_elbow":(0.67,0.47),"l_wrist":(0.31,0.65),"r_wrist":(0.69,0.65),
                   "l_hip":(0.40,0.62),"r_hip":(0.60,0.62),"l_knee":(0.39,0.80),"r_knee":(0.61,0.80),
                   "l_ankle":(0.38,0.95),"r_ankle":(0.62,0.95)}},
            {"label":"TOP - Full Curl","hl":["l_elbow","r_elbow"],"note":"~35 deg curled",
             "sk":{"head":(0.50,0.13),"l_shoulder":(0.37,0.29),"r_shoulder":(0.63,0.29),
                   "l_elbow":(0.33,0.38),"r_elbow":(0.67,0.38),"l_wrist":(0.29,0.21),"r_wrist":(0.71,0.21),
                   "l_hip":(0.40,0.62),"r_hip":(0.60,0.62),"l_knee":(0.39,0.80),"r_knee":(0.61,0.80),
                   "l_ankle":(0.38,0.95),"r_ankle":(0.62,0.95)}},
        ],
    },
    2: {
        "name":"Squat","muscle":"LEGS / GLUTES","target_reps":10,
        "key_joints":["l_knee","r_knee"],
        "rep_j":("r_hip","r_knee","r_ankle"),
        "rep_mode":"h2l","rep_start":158,"rep_end":100,
        "tips":["Feet shoulder-width","Knees track over toes","Hip crease below knee","Chest up, neutral spine"],
        "injury":{"knee_valgus":"Knee valgus! ACL at risk - push knees outward!",
                  "spine_lean":"Forward lean - disc injury risk. Chest up!"},
        "checks":[("Knee Tracking","knee_valgus",0.07),("Spine Neutral","spine_lean",0.18),
                  ("Squat Depth","squat_depth",100),("Heel Contact","heel_rise",0.04)],
        "ref":[
            {"label":"START - Standing","hl":["l_hip","r_hip"],"note":"Standing neutral",
             "sk":{"head":(0.50,0.10),"l_shoulder":(0.38,0.24),"r_shoulder":(0.62,0.24),
                   "l_elbow":(0.30,0.37),"r_elbow":(0.70,0.37),"l_wrist":(0.26,0.50),"r_wrist":(0.74,0.50),
                   "l_hip":(0.40,0.50),"r_hip":(0.60,0.50),"l_knee":(0.39,0.70),"r_knee":(0.61,0.70),
                   "l_ankle":(0.38,0.90),"r_ankle":(0.62,0.90)}},
            {"label":"BOTTOM - Below Parallel","hl":["l_knee","r_knee"],"note":"~80 deg knee",
             "sk":{"head":(0.50,0.22),"l_shoulder":(0.36,0.34),"r_shoulder":(0.64,0.34),
                   "l_elbow":(0.24,0.40),"r_elbow":(0.76,0.40),"l_wrist":(0.20,0.50),"r_wrist":(0.80,0.50),
                   "l_hip":(0.34,0.60),"r_hip":(0.66,0.60),"l_knee":(0.23,0.74),"r_knee":(0.77,0.74),
                   "l_ankle":(0.27,0.90),"r_ankle":(0.73,0.90)}},
        ],
    },
    3: {
        "name":"Push-Up","muscle":"CHEST / TRICEPS","target_reps":15,
        "key_joints":["l_elbow","r_elbow"],
        "rep_j":("r_shoulder","r_elbow","r_wrist"),
        "rep_mode":"h2l","rep_start":158,"rep_end":85,
        "tips":["Body rigid plank","Hands wider than shoulders","Elbows 45 deg from torso","Full lockout at top"],
        "injury":{"hip_sag":"Hip sagging! Lumbar compression - brace core and glutes!"},
        "checks":[("Hip Alignment","hip_sag",0.06),("Elbow Angle","arm_rom",None),
                  ("Depth","pushup_depth",85),("Neck Neutral","neck_straight",0.06)],
        "ref":[
            {"label":"TOP - Arms Extended","hl":["l_shoulder","r_shoulder"],"note":"Arms extended",
             "sk":{"head":(0.10,0.44),"l_shoulder":(0.23,0.44),"r_shoulder":(0.37,0.42),
                   "l_elbow":(0.21,0.56),"r_elbow":(0.34,0.54),"l_wrist":(0.19,0.63),"r_wrist":(0.32,0.61),
                   "l_hip":(0.57,0.44),"r_hip":(0.63,0.42),"l_knee":(0.74,0.46),"r_knee":(0.79,0.44),
                   "l_ankle":(0.89,0.50),"r_ankle":(0.93,0.48)}},
            {"label":"BOTTOM - Chest Down","hl":["l_elbow","r_elbow"],"note":"~80 deg elbow",
             "sk":{"head":(0.10,0.52),"l_shoulder":(0.23,0.52),"r_shoulder":(0.37,0.50),
                   "l_elbow":(0.19,0.63),"r_elbow":(0.32,0.61),"l_wrist":(0.17,0.68),"r_wrist":(0.30,0.66),
                   "l_hip":(0.57,0.50),"r_hip":(0.63,0.48),"l_knee":(0.74,0.50),"r_knee":(0.79,0.48),
                   "l_ankle":(0.89,0.54),"r_ankle":(0.93,0.52)}},
        ],
    },
    4: {
        "name":"Overhead Press","muscle":"SHOULDERS","target_reps":10,
        "key_joints":["l_elbow","r_elbow"],
        "rep_j":("r_shoulder","r_elbow","r_wrist"),
        "rep_mode":"l2h","rep_start":95,"rep_end":160,
        "tips":["Bar straight up","Tuck ribs brace core","Full lockout overhead","Wrists over elbows"],
        "injury":{"spine_lean":"Lumbar arch! Disc injury risk. Tuck ribs!"},
        "checks":[("Spine Neutral","spine_lean",0.12),("Full Lockout","arm_rom",None),
                  ("Wrist Align","wrist_over_elbow",0.06),("Core Braced","hip_stable",0.04)],
        "ref":[
            {"label":"RACK - Start","hl":["l_shoulder","r_shoulder"],"note":"Bar at shoulder",
             "sk":{"head":(0.50,0.11),"l_shoulder":(0.36,0.26),"r_shoulder":(0.64,0.26),
                   "l_elbow":(0.27,0.35),"r_elbow":(0.73,0.35),"l_wrist":(0.36,0.27),"r_wrist":(0.64,0.27),
                   "l_hip":(0.40,0.57),"r_hip":(0.60,0.57),"l_knee":(0.39,0.76),"r_knee":(0.61,0.76),
                   "l_ankle":(0.38,0.93),"r_ankle":(0.62,0.93)}},
            {"label":"LOCKOUT - Overhead","hl":["l_elbow","r_elbow"],"note":"~175 deg lockout",
             "sk":{"head":(0.50,0.16),"l_shoulder":(0.38,0.30),"r_shoulder":(0.62,0.30),
                   "l_elbow":(0.36,0.14),"r_elbow":(0.64,0.14),"l_wrist":(0.37,0.04),"r_wrist":(0.63,0.04),
                   "l_hip":(0.40,0.57),"r_hip":(0.60,0.57),"l_knee":(0.39,0.76),"r_knee":(0.61,0.76),
                   "l_ankle":(0.38,0.93),"r_ankle":(0.62,0.93)}},
        ],
    },
    5: {
        "name":"Lateral Raise","muscle":"SIDE DELTS","target_reps":12,
        "key_joints":["l_shoulder","r_shoulder"],
        "rep_j":("l_hip","l_shoulder","l_elbow"),
        "rep_mode":"l2h","rep_start":30,"rep_end":75,
        "tips":["Soft elbows throughout","Raise to shoulder height only","No shrugging","Thumbs slightly down"],
        "injury":{"shrugging":"Shoulder shrugging! Supraspinatus impingement. Relax traps!"},
        "checks":[("Arm Level","lateral_level",0.06),("No Shrugging","shoulder_shrug",0.05),
                  ("Soft Elbows","arm_rom",None),("Control","shoulder_stable",0.06)],
        "ref":[
            {"label":"START - Arms at Sides","hl":["l_wrist","r_wrist"],"note":"Arms hanging neutral",
             "sk":{"head":(0.50,0.11),"l_shoulder":(0.38,0.27),"r_shoulder":(0.62,0.27),
                   "l_elbow":(0.32,0.43),"r_elbow":(0.68,0.43),"l_wrist":(0.30,0.59),"r_wrist":(0.70,0.59),
                   "l_hip":(0.40,0.57),"r_hip":(0.60,0.57),"l_knee":(0.39,0.76),"r_knee":(0.61,0.76),
                   "l_ankle":(0.38,0.93),"r_ankle":(0.62,0.93)}},
            {"label":"TOP - Shoulder Height","hl":["l_elbow","r_elbow"],"note":"Arms at shoulder height",
             "sk":{"head":(0.50,0.11),"l_shoulder":(0.38,0.27),"r_shoulder":(0.62,0.27),
                   "l_elbow":(0.17,0.27),"r_elbow":(0.83,0.27),"l_wrist":(0.06,0.29),"r_wrist":(0.94,0.29),
                   "l_hip":(0.40,0.57),"r_hip":(0.60,0.57),"l_knee":(0.39,0.76),"r_knee":(0.61,0.76),
                   "l_ankle":(0.38,0.93),"r_ankle":(0.62,0.93)}},
        ],
    },
    6: {
        "name":"Tricep Extension","muscle":"TRICEPS","target_reps":12,
        "key_joints":["l_elbow","r_elbow"],
        "rep_j":("r_shoulder","r_elbow","r_wrist"),
        "rep_mode":"l2h","rep_start":80,"rep_end":160,
        "tips":["Upper arms pinned","Only forearms move","Full extension","Slow eccentric"],
        "injury":{"moving_upper_arm":"Upper arm moving! Less isolation and shoulder strain."},
        "checks":[("Upper Arm Still","upper_arm_stable",0.06),("Full Extension","arm_rom",None),
                  ("Wrist Neutral","wrist_straight",0.09),("Elbow Position","elbow_high",None)],
        "ref":[
            {"label":"TOP - Arms Overhead","hl":["l_elbow","r_elbow"],"note":"~80 deg behind head",
             "sk":{"head":(0.50,0.10),"l_shoulder":(0.38,0.24),"r_shoulder":(0.62,0.24),
                   "l_elbow":(0.35,0.14),"r_elbow":(0.65,0.14),"l_wrist":(0.31,0.30),"r_wrist":(0.69,0.30),
                   "l_hip":(0.40,0.57),"r_hip":(0.60,0.57),"l_knee":(0.39,0.76),"r_knee":(0.61,0.76),
                   "l_ankle":(0.38,0.93),"r_ankle":(0.62,0.93)}},
            {"label":"BOTTOM - Extended","hl":["l_wrist","r_wrist"],"note":"~170 deg extended",
             "sk":{"head":(0.50,0.10),"l_shoulder":(0.38,0.24),"r_shoulder":(0.62,0.24),
                   "l_elbow":(0.34,0.13),"r_elbow":(0.66,0.13),"l_wrist":(0.34,0.02),"r_wrist":(0.66,0.02),
                   "l_hip":(0.40,0.57),"r_hip":(0.60,0.57),"l_knee":(0.39,0.76),"r_knee":(0.61,0.76),
                   "l_ankle":(0.38,0.93),"r_ankle":(0.62,0.93)}},
        ],
    },
    7: {
        "name":"Deadlift","muscle":"BACK / HAMSTRINGS","target_reps":8,
        "key_joints":["l_hip","r_hip"],
        "rep_j":("l_shoulder","l_hip","l_knee"),
        "rep_mode":"l2h","rep_start":100,"rep_end":162,
        "tips":["NEVER round lower back","Push hips back not down","Bar close to legs","Eyes forward"],
        "injury":{"back_round":"BACK ROUNDING! HIGH disc herniation risk! Chest up!"},
        "checks":[("Spine Neutral","spine_lean",0.18),("Hip Hinge","hip_hinge_ok",None),
                  ("Bar Path","shoulder_stack",0.09),("Soft Knees","knee_slight",None)],
        "ref":[
            {"label":"TOP - Standing Lockout","hl":["l_hip","r_hip"],"note":"Full hip extension",
             "sk":{"head":(0.50,0.10),"l_shoulder":(0.38,0.24),"r_shoulder":(0.62,0.24),
                   "l_elbow":(0.32,0.38),"r_elbow":(0.68,0.38),"l_wrist":(0.38,0.52),"r_wrist":(0.62,0.52),
                   "l_hip":(0.40,0.52),"r_hip":(0.60,0.52),"l_knee":(0.39,0.70),"r_knee":(0.61,0.70),
                   "l_ankle":(0.38,0.90),"r_ankle":(0.62,0.90)}},
            {"label":"BOTTOM - Hip Hinged","hl":["l_hip","r_hip"],"note":"Hips back, neutral spine",
             "sk":{"head":(0.46,0.30),"l_shoulder":(0.32,0.38),"r_shoulder":(0.54,0.36),
                   "l_elbow":(0.33,0.52),"r_elbow":(0.55,0.50),"l_wrist":(0.37,0.64),"r_wrist":(0.59,0.62),
                   "l_hip":(0.44,0.56),"r_hip":(0.62,0.54),"l_knee":(0.40,0.72),"r_knee":(0.60,0.70),
                   "l_ankle":(0.38,0.90),"r_ankle":(0.60,0.90)}},
        ],
    },
    8: {
        "name":"Plank Hold","muscle":"CORE / ABS","target_reps":3,
        "key_joints":["l_hip","r_hip"],
        "rep_j":("l_shoulder","l_hip","l_ankle"),
        "rep_mode":"hold","rep_start":160,"rep_end":160,
        "tips":["Head to heel rigid line","Shoulders over wrists","Squeeze glutes AND core","Breathe steadily"],
        "injury":{"hip_sag":"Hip sagging! Lumbar compression - raise hips to neutral.",
                  "hip_high":"Hips too high - reduces core activation, lower them."},
        "checks":[("Hip Level","hip_plank",0.05),("Shoulder Stack","shoulder_wrist",0.07),
                  ("Head Neutral","neck_straight",0.06),("Core Tight","hip_stable",0.03)],
        "ref":[
            {"label":"PLANK - Perfect Position","hl":["l_hip","r_hip"],"note":"Body straight ~180 deg",
             "sk":{"head":(0.09,0.45),"l_shoulder":(0.21,0.45),"r_shoulder":(0.29,0.43),
                   "l_elbow":(0.19,0.55),"r_elbow":(0.27,0.53),"l_wrist":(0.17,0.60),"r_wrist":(0.25,0.58),
                   "l_hip":(0.54,0.44),"r_hip":(0.60,0.43),"l_knee":(0.71,0.46),"r_knee":(0.77,0.45),
                   "l_ankle":(0.87,0.50),"r_ankle":(0.92,0.49)}},
        ],
    },
}
def ang3(a,b,c):
    ax,ay=a[0]-b[0],a[1]-b[1]; cx,cy=c[0]-b[0],c[1]-b[1]
    d=ax*cx+ay*cy; m=max(math.hypot(ax,ay)*math.hypot(cx,cy),1e-6)
    return math.degrees(math.acos(max(-1.0,min(1.0,d/m))))
def gpx(lms,k,W,H,vis=0.28):
    i=LM.get(k)
    if i is None: return None
    p=lms[i]; return (int(p.x*W),int(p.y*H)) if p.visibility>vis else None
def gnm(lms,k,vis=0.28):
    i=LM.get(k)
    if i is None: return None
    p=lms[i]; return (p.x,p.y) if p.visibility>vis else None
class RepCounter:
    def __init__(self): self.reset()
    def reset(self):
        self.count=0; self.state="neutral"
        self.hold_start=0.0; self.hold_done=False
    def update(self, angle, ex):
        mode=ex["rep_mode"]
        if mode=="hold":
            if angle is not None and angle>155:
                if self.hold_start==0.0: self.hold_start=time.time()
                elif time.time()-self.hold_start>2.0 and not self.hold_done:
                    self.count+=1; self.hold_done=True; return True
            else:
                self.hold_start=0.0; self.hold_done=False
            return False
        if angle is None: return False
        s,e=ex["rep_start"],ex["rep_end"]
        if mode=="h2l":  # high-to-low: e.g. bicep curl — arm goes extended(hi) → curled(lo)
            if self.state=="neutral" and angle>=s: self.state="at_start"
            elif self.state=="at_start" and angle<=e: self.state="at_end"
            elif self.state=="at_end" and angle>=s:
                self.state="at_start"; self.count+=1; return True
        elif mode=="l2h":  # low-to-high: e.g. overhead press — starts low → extends high
            if self.state=="neutral" and angle<=s: self.state="at_start"
            elif self.state=="at_start" and angle>=e: self.state="at_end"
            elif self.state=="at_end" and angle<=s:
                self.state="at_start"; self.count+=1; return True
        return False
def analyse(lms, ex, W, H):
    name=ex["name"]; errs=0; feedback=[]; quality={}; angles={}; injury=None
    def pt(k): return gpx(lms,k,W,H)
    def nm(k): return gnm(lms,k)
    if name=="Bicep Curl":
        for sd,lb in [("l","L"),("r","R")]:
            s,e,w=pt(f"{sd}_shoulder"),pt(f"{sd}_elbow"),pt(f"{sd}_wrist")
            if all([s,e,w]): angles[f"{lb} Elbow"]=ang3(s,e,w)
        ls_n,le_n=nm("l_shoulder"),nm("l_elbow")
        rs_n,re_n=nm("r_shoulder"),nm("r_elbow")
        drift=0
        if ls_n and le_n: drift=max(drift,abs(le_n[0]-ls_n[0]))
        if rs_n and re_n: drift=max(drift,abs(re_n[0]-rs_n[0]))
        q=max(0,int((1-drift/0.18)*100)); quality["Elbow Stability"]=q
        if drift>0.14:
            errs+=1; feedback.append(("Elbows drifting! Keep at sides.",True))
            injury=ex["injury"].get("elbow_drift")
        else: feedback.append(("Elbows stable",False))
        lw_n,le_n2=nm("l_wrist"),nm("l_elbow")
        if lw_n and le_n2:
            wb=abs(lw_n[0]-le_n2[0]); quality["Wrist Neutral"]=max(0,int((1-wb/0.12)*100))
            if wb>0.09: errs+=1; feedback.append(("Keep wrists straight!",True))
            else: feedback.append(("Wrists neutral",False))
        quality.setdefault("Range of Motion",80); quality.setdefault("Shoulder Still",80)
    elif name=="Squat":
        for sd in ["l","r"]:
            h,k,a=pt(f"{sd}_hip"),pt(f"{sd}_knee"),pt(f"{sd}_ankle")
            if all([h,k,a]): angles[f"{sd.upper()} Knee"]=ang3(h,k,a)
        lk_n,la_n=nm("l_knee"),nm("l_ankle"); rk_n,ra_n=nm("r_knee"),nm("r_ankle")
        valgus=False; vscore=100
        for kn,an in [(lk_n,la_n),(rk_n,ra_n)]:
            if kn and an:
                d=abs(kn[0]-an[0]); vscore=min(vscore,max(0,int((1-d/0.10)*100)))
                if d>0.08: valgus=True
        quality["Knee Tracking"]=vscore
        if valgus:
            errs+=1; feedback.append(("Knees caving in! Push out.",True))
            injury=ex["injury"].get("knee_valgus")
        else: feedback.append(("Knee tracking good",False))
        ls_n,lh_n,lk_n2=nm("l_shoulder"),nm("l_hip"),nm("l_knee")
        if ls_n and lh_n and lk_n2:
            lh=abs(lh_n[1]-lk_n2[1]) or 0.1
            lean=abs(ls_n[0]-lh_n[0])/lh; quality["Spine Neutral"]=max(0,int((1-lean/0.4)*100))
            if lean>0.25: errs+=1; feedback.append(("Lean forward — chest up!",True)); injury=ex["injury"].get("spine_lean")
            else: feedback.append(("Spine neutral",False))
        quality.setdefault("Squat Depth",70); quality.setdefault("Heel Contact",85)
    elif name=="Push-Up":
        for sd,lb in [("l","L"),("r","R")]:
            s,e,w=pt(f"{sd}_shoulder"),pt(f"{sd}_elbow"),pt(f"{sd}_wrist")
            if all([s,e,w]): angles[f"{lb} Elbow"]=ang3(s,e,w)
        ls_n,lh_n,la_n=nm("l_shoulder"),nm("l_hip"),nm("l_ankle")
        if ls_n and lh_n and la_n:
            t=(lh_n[0]-ls_n[0])/max(la_n[0]-ls_n[0],0.01)
            ey=ls_n[1]+t*(la_n[1]-ls_n[1]); sag=lh_n[1]-ey
            quality["Hip Alignment"]=max(0,int((1-abs(sag)/0.08)*100))
            if sag>0.07: errs+=1; feedback.append(("Hips sagging! Brace core.",True)); injury=ex["injury"].get("hip_sag")
            elif sag<-0.07: errs+=1; feedback.append(("Hips too high (piking)!",True))
            else: feedback.append(("Body alignment good",False))
        quality.setdefault("Elbow Angle",75); quality.setdefault("Depth",70); quality.setdefault("Neck Neutral",80)
    elif name=="Overhead Press":
        for sd,lb in [("l","L"),("r","R")]:
            s,e,w=pt(f"{sd}_shoulder"),pt(f"{sd}_elbow"),pt(f"{sd}_wrist")
            if all([s,e,w]): angles[f"{lb} Elbow"]=ang3(s,e,w)
        ls_n,lh_n,lk_n=nm("l_shoulder"),nm("l_hip"),nm("l_knee")
        if ls_n and lh_n and lk_n:
            lh=abs(lh_n[1]-lk_n[1]) or 0.1; lean=abs(ls_n[0]-lh_n[0])/lh
            quality["Spine Neutral"]=max(0,int((1-lean/0.3)*100))
            if lean>0.18: errs+=1; feedback.append(("Arch in back — tuck ribs!",True)); injury=ex["injury"].get("spine_lean")
            else: feedback.append(("Spine neutral",False))
        quality.setdefault("Full Lockout",75); quality.setdefault("Wrist Align",80); quality.setdefault("Core Braced",80)
    elif name=="Lateral Raise":
        ls_n,rs_n=nm("l_shoulder"),nm("r_shoulder")
        le_n,re_n=nm("l_elbow"),nm("r_elbow")
        if all([ls_n,rs_n,le_n,re_n]):
            diff=abs((le_n[1]-ls_n[1])-(re_n[1]-rs_n[1]))
            quality["Arm Level"]=max(0,int((1-diff/0.10)*100))
            if diff>0.07: errs+=1; feedback.append(("Arms uneven — raise both equally.",True))
            else: feedback.append(("Arms level",False))
        ls_n2=nm("l_shoulder")
        if ls_n2 and lms[7].visibility>0.3:
            dist=abs(ls_n2[1]-lms[7].y)
            quality["No Shrugging"]=max(0,int((dist/0.15)*100))
            if dist<0.08: errs+=1; feedback.append(("Shoulder shrugging! Relax traps.",True)); injury=ex["injury"].get("shrugging")
            else: feedback.append(("No shrugging",False))
        quality.setdefault("Soft Elbows",80); quality.setdefault("Control",75)
    elif name=="Tricep Extension":
        for sd,lb in [("l","L"),("r","R")]:
            s,e,w=pt(f"{sd}_shoulder"),pt(f"{sd}_elbow"),pt(f"{sd}_wrist")
            if all([s,e,w]): angles[f"{lb} Elbow"]=ang3(s,e,w)
        le_n,ls_n=nm("l_elbow"),nm("l_shoulder")
        if le_n and ls_n:
            d=abs(le_n[0]-ls_n[0]); quality["Upper Arm Still"]=max(0,int((1-d/0.10)*100))
            if d>0.08: errs+=1; feedback.append(("Upper arms moving! Keep fixed.",True)); injury=ex["injury"].get("moving_upper_arm")
            else: feedback.append(("Upper arms stable",False))
        quality.setdefault("Full Extension",75); quality.setdefault("Wrist Neutral",80); quality.setdefault("Elbow Position",80)
    elif name=="Deadlift":
        ls,lh,lk=pt("l_shoulder"),pt("l_hip"),pt("l_knee")
        if all([ls,lh,lk]): angles["Hip"]=ang3(ls,lh,lk)
        ls_n,lh_n=nm("l_shoulder"),nm("l_hip")
        if ls_n and lh_n:
            horiz=abs(ls_n[0]-lh_n[0]); vert=abs(ls_n[1]-lh_n[1]) or 0.05
            rr=horiz/vert; quality["Spine Neutral"]=max(0,int((1-rr/0.6)*100))
            if rr>0.45: errs+=1; feedback.append(("BACK ROUNDING! Chest up NOW!",True)); injury=ex["injury"].get("back_round")
            else: feedback.append(("Spine neutral",False))
        quality.setdefault("Hip Hinge",75); quality.setdefault("Bar Path",80); quality.setdefault("Soft Knees",80)
    elif name=="Plank Hold":
        ls,lh,la=pt("l_shoulder"),pt("l_hip"),pt("l_ankle")
        if all([ls,lh,la]):
            a=ang3(ls,lh,la); angles["Body"]=a
            quality["Hip Level"]=max(0,int((1-abs(a-178)/20)*100))
            if a<158: errs+=1; feedback.append(("Hips sagging! Raise hips.",True)); injury=ex["injury"].get("hip_sag")
            elif a>202: errs+=1; feedback.append(("Hips too high! Lower them.",True)); injury=ex["injury"].get("hip_high")
            else: feedback.append(("Body alignment perfect",False))
        ls_n,lw_n=nm("l_shoulder"),nm("l_wrist")
        if ls_n and lw_n:
            d=abs(ls_n[0]-lw_n[0]); quality["Shoulder Stack"]=max(0,int((1-d/0.10)*100))
            if d>0.08: errs+=1; feedback.append(("Shoulders over wrists!",True))
            else: feedback.append(("Shoulders stacked",False))
        quality.setdefault("Head Neutral",80); quality.setdefault("Core Tight",80)
    else:
        feedback.append(("Tracking...",False))
        for c in ex["checks"]: quality.setdefault(c[0],75)
    n=max(len(ex["checks"]),1); w=100.0/n
    score=max(0,min(100,int(sum(quality.get(c[0],70)*w for c in ex["checks"])/100)))
    ok=errs==0 and score>=55
    if not feedback: feedback.append(("Analysing...",False))
    return ok, score, feedback[:4], quality, angles, injury
CG=(50,220,50); CR=(50,50,235); CC=(220,180,0); CW=(240,240,245)
CY=(0,210,240); CO=(0,155,240); CGR=(100,100,120); CD=(15,15,22)
CP=(22,22,35);  CB2=(18,18,30); CBK=(0,0,0)
def t(img,text,pos,sc=0.6,col=CW,th=1,bg=True,bgc=None,fn=cv2.FONT_HERSHEY_SIMPLEX):
    x,y=pos; (tw,tth),_=cv2.getTextSize(text,fn,sc,th)
    if bg: cv2.rectangle(img,(x-3,y-tth-4),(x+tw+3,y+5),bgc if bgc else CD,-1)
    cv2.putText(img,text,(x,y),fn,sc,col,th,cv2.LINE_AA)
def b(img,x,y,w,h,pct,col,bg=(40,40,55)):
    cv2.rectangle(img,(x,y),(x+w,y+h),bg,-1)
    if pct>0: cv2.rectangle(img,(x,y),(x+int(w*pct/100),y+h),col,-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),(60,60,80),1)
def draw_ref(canvas,pd):
    H2,W2=canvas.shape[:2]; sk=pd["sk"]; hl=pd.get("hl",[])
    def px(k): x,y=sk[k]; return int(x*W2),int(y*H2)
    for a,bb in CONN:
        if a in sk and bb in sk:
            is_hl=a in hl or bb in hl
            cv2.line(canvas,px(a),px(bb),(0,220,100) if is_hl else CC,4 if is_hl else 2,cv2.LINE_AA)
    for k in sk:
        p=px(k); r=9 if k in hl else 6; c=(0,220,100) if k in hl else CC
        cv2.circle(canvas,p,r,c,-1,cv2.LINE_AA); cv2.circle(canvas,p,r+2,CBK,1,cv2.LINE_AA)
    if "head" in sk: hx,hy=px("head"); cv2.circle(canvas,(hx,hy),18,CC,2,cv2.LINE_AA)
    note=pd.get("note","")
    if note: t(canvas,note,(8,H2-14),0.45,CY,1,True,(0,0,0))
def main():
    mp_pose=mp.solutions.pose
    pe=mp_pose.Pose(model_complexity=1,smooth_landmarks=True,
                    min_detection_confidence=0.50,min_tracking_confidence=0.50)
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): cap=cv2.VideoCapture(1)
    if not cap.isOpened(): print("No camera!"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480); cap.set(cv2.CAP_PROP_FPS,30)
    CW2,CH=640,480; RW,RH=380,480; PW=290; TW=CW2+RW+PW; TH=CH+120
    WIN="AI GYM TRAINER  |  1-8:Exercise  V:Voice  R:Reset  Q:Quit"
    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN,TW,TH)
    cur=1; ex=EXERCISES[cur]; rc=RepCounter(); sets=1
    von=False; fscore=0; iscor=False; imsg=""; iend=0.0
    rst=0; rtm=time.time(); ri=2.5; fb=[]; ql={}; agls={}
    ft=time.time(); fc=0; fd=0; lfs=0.0
    print("\n=== AI GYM TRAINER v2 — ALL BUGS FIXED ===")
    print("Keys: 1-8 exercise | V voice | R reset | Q quit\n")
    while True:
        ret,frame=cap.read()
        if not ret: frame=np.zeros((CH,CW2,3),np.uint8)
        frame=cv2.flip(frame,1); H,W=frame.shape[:2]; now=time.time()
        fc+=1
        if now-ft>=1.0: fd=fc; fc=0; ft=now
        if now-rtm>ri:
            rtm=now
            poses=ex.get("ref",[])
            if poses: rst=(rst+1)%len(poses)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
        res=pe.process(rgb); rgb.flags.writeable=True
        has=False; ainj=None
        if res.pose_landmarks:
            has=True; lms=res.pose_landmarks.landmark
            iscor,fscore,fb,ql,agls,ainj=analyse(lms,ex,W,H)
            if ainj: imsg=ainj; iend=now+5.0
            rj=ex["rep_j"]; ap=gpx(lms,rj[0],W,H); bp=gpx(lms,rj[1],W,H); cp2=gpx(lms,rj[2],W,H)
            ra=ang3(ap,bp,cp2) if all([ap,bp,cp2]) else None
            if rc.update(ra,ex):
                n=rc.count; print(f"[Rep] {ex['name']} — {n}")
                if von:
                    if n%5==0: speak(f"{n} reps! Great work!")
                    else: speak(f"Rep {n}")
                if n>=ex["target_reps"]:
                    sets+=1; rc.reset(); print(f"[Set] Set {sets-1} complete!")
                    if von: speak("Set complete! Take a rest.")
            if von and now-lfs>5.0:
                errs=[m for m,e in fb if e]
                if errs: speak(errs[0]); lfs=now
                elif iscor and now-lfs>10.0: speak("Good form! Keep going."); lfs=now
            sc2=CG if iscor else CR
            for ak,bk in CONN:
                ai,bi=LM.get(ak),LM.get(bk)
                if ai is None or bi is None: continue
                pa,pb=lms[ai],lms[bi]
                if pa.visibility>0.28 and pb.visibility>0.28:
                    cv2.line(frame,(int(pa.x*W),int(pa.y*H)),(int(pb.x*W),int(pb.y*H)),sc2,3,cv2.LINE_AA)
            for key,idx in LM.items():
                if key=="nose": continue
                p=lms[idx]
                if p.visibility>0.28:
                    px2=(int(p.x*W),int(p.y*H)); isk=key in ex["key_joints"]
                    cv2.circle(frame,px2,9 if isk else 6,sc2,-1,cv2.LINE_AA)
                    cv2.circle(frame,px2,(9 if isk else 6)+2,CBK,1,cv2.LINE_AA)
            ns=lms[LM["nose"]]
            if ns.visibility>0.28: cv2.circle(frame,(int(ns.x*W),int(ns.y*H)),15,sc2,2,cv2.LINE_AA)
            # ANGLE LABELS
            jmap={"L Elbow":"l_elbow","R Elbow":"r_elbow","L Knee":"l_knee","R Knee":"r_knee",
                  "Hip":"l_hip","R Hip":"r_hip","Body":"l_hip","Elbow":"r_elbow"}
            for jn,ag in agls.items():
                mk=jmap.get(jn)
                if mk:
                    p2=gpx(lms,mk,W,H)
                    if p2:
                        c2=CG if 40<=ag<=175 else CO
                        t(frame,f"{int(ag)}",(p2[0]+10,p2[1]-6),0.45,c2,1)
        fn2=cv2.FONT_HERSHEY_SIMPLEX
        if has:
            lbl="Correct Posture" if iscor else "Incorrect Posture"
            col2=(0,230,0) if iscor else (0,0,220)
            cv2.putText(frame,lbl,(19,44),fn2,1.1,CBK,4,cv2.LINE_AA)
            cv2.putText(frame,lbl,(17,42),fn2,1.1,col2,2,cv2.LINE_AA)
        else:
            t(frame,"Stand in frame...",(15,42),0.9,CY,2,True,CD)
        t(frame,f"FPS:{fd}",(W-82,20),0.42,CGR,1,False)
        rc2=np.full((RH,RW,3),CB2,np.uint8)
        cv2.rectangle(rc2,(0,0),(RW,48),(28,28,46),-1)
        t(rc2,"CORRECT FORM DEMO",(10,32),0.60,CC,2,False)
        poses=ex.get("ref",[])
        if poses:
            pd2=poses[rst%len(poses)]; area=rc2[52:]
            draw_ref(area,pd2); t(rc2,pd2["label"],(8,RH-14),0.44,CY,1,True,(0,0,0))
            for i in range(len(poses)):
                cv2.circle(rc2,(RW-20-(len(poses)-1-i)*14,38),5,CG if i==rst else CGR,-1)
        ty2=56
        for tip in ex["tips"][:3]:
            t(rc2,f"* {tip[:38]}",(6,ty2),0.37,(180,200,255),1,False); ty2+=17
        panel=np.full((TH,PW,3),CP,np.uint8)
        cv2.rectangle(panel,(0,0),(PW,52),(30,30,52),-1)
        t(panel,ex["name"],(8,28),0.65,CW,2,False)
        t(panel,ex["muscle"],(8,46),0.40,CC,1,False)
        py=60
        cv2.rectangle(panel,(6,py),(PW-6,py+66),(28,28,46),-1); cv2.rectangle(panel,(6,py),(PW-6,py+66),(50,50,70),1)
        t(panel,"REPS",(14,py+16),0.42,CGR,1,False)
        t(panel,str(rc.count),(14,py+52),1.2,CG,2,False)
        t(panel,f"/ {ex['target_reps']}",(68,py+52),0.70,CGR,1,False)
        t(panel,f"SET {sets}",(PW-76,py+38),0.55,CY,1,False)
        py+=74
        if has and ex["rep_mode"]!="hold":
            sc3=CG if rc.state=="at_start" else (CO if rc.state=="at_end" else CGR)
            t(panel,f"State: {rc.state}",(8,py+12),0.38,sc3,1,False)
        py+=18
        sc4=CG if fscore>=75 else (CO if fscore>=50 else CR)
        t(panel,f"FORM  {fscore}%",(8,py+14),0.50,sc4,1,False); b(panel,8,py+18,PW-16,10,fscore,sc4); py+=36
        vc=( (0,200,255) if von else CGR)
        cv2.rectangle(panel,(6,py),(PW-6,py+22),(28,28,46),-1)
        t(panel,("[V] VOICE ON" if von else "[V] VOICE OFF"),(10,py+15),0.48,vc,1,False); py+=28
        t(panel,"QUALITY",(8,py+12),0.40,CGR,1,False); py+=16
        for ck in ex["checks"]:
            qv=ql.get(ck[0],0); qc=CG if qv>=75 else (CO if qv>=50 else CR)
            t(panel,ck[0][:18],(8,py+12),0.36,CW,1,False)
            b(panel,8,py+15,PW-52,7,qv,qc); t(panel,f"{qv}%",(PW-42,py+14),0.36,qc,1,False); py+=26
            if py>TH-100: break
        py+=4
        t(panel,"FEEDBACK",(8,py+12),0.40,CGR,1,False); py+=16
        for msg,ie in fb[:4]:
            mc=CR if ie else CG; t(panel,("X " if ie else "OK ")[:2]+msg[:28],(8,py+12),0.36,mc,1,False); py+=18
            if py>TH-70: break
        if now<iend and int(now*3)%2==0:
            iy=min(py+10,TH-80)
            cv2.rectangle(panel,(4,iy-18),(PW-4,iy+55),(55,15,15),-1); cv2.rectangle(panel,(4,iy-18),(PW-4,iy+55),CR,1)
            for ln in [imsg[i:i+28] for i in range(0,min(len(imsg),56),28)]:
                t(panel,ln,(8,iy),0.36,CR,1,False); iy+=18
        bot=np.full((120,CW2+RW,3),(18,18,28),np.uint8)
        cv2.line(bot,(0,0),(CW2+RW,0),(50,50,70),1)
        bx2=6; bw2=(CW2+RW-12)//8
        for eid2 in range(1,9):
            ed=EXERCISES[eid2]; act=(eid2==cur)
            cv2.rectangle(bot,(bx2,8),(bx2+bw2-4,112),(40,70,40) if act else (25,25,38),-1)
            if act: cv2.rectangle(bot,(bx2,8),(bx2+bw2-4,112),CG,1)
            t(bot,f"[{eid2}]",(bx2+3,24),0.40,CG if act else CGR,1,False)
            for i2,w2 in enumerate(ed["name"].split()[:3]):
                t(bot,w2[:10],(bx2+3,42+i2*17),0.38,CW if act else (130,130,150),1,False)
            t(bot,ed["muscle"][:12],(bx2+3,100),0.30,CC if act else CGR,1,False)
            bx2+=bw2
        cf=np.zeros((TH,CW2,3),np.uint8); cf[:CH]=frame; cf[CH:]=bot[:,:CW2]
        rf=np.zeros((TH,RW,3),np.uint8); rf[:RH]=rc2; rf[RH:]=bot[:,CW2:]
        full=np.hstack([cf,rf,panel])
        cv2.line(full,(CW2,0),(CW2,TH),(50,50,70),1); cv2.line(full,(CW2+RW,0),(CW2+RW,TH),(50,50,70),1)
        cv2.imshow(WIN,full)
        key=cv2.waitKey(1)&0xFF
        if key in (ord('q'),ord('Q'),27): break
        elif key in (ord('v'),ord('V')):
            von=not von; print(f"[Voice] {'ON' if von else 'OFF'}")
            if von and HAS_TTS: speak(f"Voice on. {ex['name']}.")
            elif von and not HAS_TTS: print("[Voice] No TTS — install pyttsx3 or espeak")
        elif key in (ord('r'),ord('R')):
            rc.reset(); sets=1; print("[Reset] Done.")
        elif ord('1')<=key<=ord('8'):
            eid3=key-ord('0')
            if eid3 in EXERCISES:
                cur=eid3; ex=EXERCISES[cur]; rc.reset(); sets=1; rst=0
                print(f"[Exercise] -> {ex['name']}")
                if von and HAS_TTS: speak(f"{ex['name']}.")
    cap.release(); pe.close(); cv2.destroyAllWindows(); print("[Done]")
if __name__=="__main__": main()
