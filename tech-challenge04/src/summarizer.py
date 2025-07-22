# summarizer.py
import json
from collections import Counter, defaultdict

class Summarizer:
    def __init__(self):
        self.emotions = Counter()
        self.activities = Counter()
        self.anomalies = 0
        self.total_frames = 0
        self.names = Counter()
        self.name_emotions = defaultdict(list) 
        self.name_activities = defaultdict(list) 

    # summarizer.py
    def update(self, emotion, activity, anomaly, names=None, name_emotions=None, name_activities=None):
        self.emotions[emotion] += 1
    
        # Corrija aqui:
        if isinstance(activity, dict):
            # Exemplo: concatene os valores dos campos principais
            activity_str = f"{activity.get('movement', 'Unknown')}, {activity.get('reaction', '')}"
        else:
            activity_str = str(activity)
        self.activities[activity_str] += 1
    
        if anomaly:
            self.anomalies += 1
        self.total_frames += 1
        if names:
            self.names.update(names)
        if name_emotions:
            for name, emo in name_emotions.items():
                self.name_emotions[name].append(emo)
        if name_activities:
            for name, act in name_activities.items():
                self.name_activities[name].append(act)

    def save(self, path):
        summary = {
            "total_frames": self.total_frames,
            "emotions": dict(self.emotions),
            "activities": dict(self.activities),
            "anomalies": self.anomalies,
            "names": dict(self.names),
            "name_emotions": dict(self.name_emotions),
            "name_activities": dict(self.name_activities),
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)