# Path: Mastermind_Finder_Bot/src/outreach_engine/OutreachGenerator.py
# Generates personalized outreach messages based on profile info

class OutreachGenerator:
    def generate_message(self, template, candidate):
        \"\"\"Generates a personalized outreach message for a candidate.\"\"\"
        return template.format(name=candidate['name'], field=candidate['field'])
