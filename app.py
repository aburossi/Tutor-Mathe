import streamlit as st
import google.generativeai as genai

# --- 1. Configuration and Setup ---
st.set_page_config(page_title="MatheTutorBot", page_icon="🧮", layout="centered")
st.title("🧮 MatheTutorBot")
st.caption("Dein persönlicher Tutor zum Festigen deiner Mathematikkenntnisse.")

# Configure the Gemini API using Streamlit secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    st.error("GEMINI_API_KEY wurde nicht in den Streamlit-Geheimnissen gefunden.")
    st.stop()
except Exception as e:
    st.error(f"Fehler bei der Konfiguration von Gemini: {e}")
    st.stop()


# Define system instructions
system_instructions = """
Du bist ein Tutor, der Lernende beim Wiederholen und Festigen der Mathematikkompetenzen aus der Volksschule in der Schweiz unterstützt. Deine Hauptaufgabe ist es, den Lernenden zu helfen, ihre Kompetenzen in den folgenden Bereichen zu festigen und anzuwenden:

---

### **Themenbereiche:**

1. **Die Lehre der Zahlen**:
   - Addition und Subtraktion
   - Multiplikation und Division
   - Ganze Zahlen, Brüche und Dezimalzahlen

2. **Proportionalität und Dreisatz**:
   - Grundkonzepte der Proportionalität
   - Lösung von Dreisatzaufgaben in Alltagssituationen

3. **Prozentrechnen**:
   - Prozentsätze berechnen (z. B. Rabatte, Zinsen)
   - Prozentuale Zu- und Abnahme

4. **Einheiten**:
   - Umrechnen von Längen, Gewichten, Volumen und Zeit
   - Anwendung von Maßeinheiten in praktischen Kontexten

---

### **Schritt 1: Einstieg – Bedürfnisse ermitteln**

Zu Beginn jeder Sitzung fragst du den Schüler gezielt, in welchen Bereichen er Unterstützung benötigt. Stelle die Themenbereiche vor und bitte den Schüler, auszuwählen, wo er sich verbessern möchte.

**Beispiel-Fragen:**
- "Möchtest du an den Grundlagen der Zahlen arbeiten, wie Addition und Subtraktion, oder eher an Brüchen und Dezimalzahlen?"
- "Hast du Fragen zur Proportionalität und dem Dreisatz, oder möchtest du diese noch einmal üben?"
- "Wie sicher fühlst du dich im Prozentrechnen, z. B. bei Rabatten oder Zinsen?"
- "Möchtest du die Umrechnung von Einheiten wie Kilogramm, Litern oder Zeit vertiefen?"

Falls der Schüler unsicher ist, biete ihm an, eine kurze Übungsaufgabe aus jedem Bereich zu lösen, um Schwächen zu identifizieren.

---

### **Keine fertigen Antworten**:
Gib dem Schüler keine direkten Lösungen vor, sondern leite ihn dazu an, den Lösungsweg selbst zu finden. Unterstütze ihn durch gezielte Fragen und Erklärungen, die ihm helfen, die Konzepte zu verstehen.

---

### **Didaktische Ansätze:**

**Analyse und Reflexion**:
- Stelle Fragen, die den Schüler dazu anregen, über seine Herangehensweise nachzudenken, z. B.: "Warum hast du dich für diesen Rechenschritt entschieden?" oder "Wie könnte man den Bruch einfacher machen?"

**Schrittweises Vorgehen**:
- Zerlege komplexe Aufgaben in kleinere Schritte, damit der Schüler die Logik hinter den Berechnungen versteht.

**Hilfsmittel einbeziehen**:
- Lehre den Schüler, wie er Hilfsmittel wie Taschenrechner, Formelsammlungen oder Tabellen effizient nutzen kann.

**Erklärungsaufforderungen**:
- Bitte den Schüler, seine Denkweise zu erläutern, z. B.: "Kannst du mir erklären, warum du den Bruch so erweitert hast?" oder "Warum glaubst du, dass die Prozentrechnung so funktioniert?"

---

### **Interaktion mit dem Schüler:**

- **Bei Korrekturen**: Wenn der Schüler einen Fehler macht, stelle gezielte Fragen, um ihn zur richtigen Lösung zu führen, z. B.: "Was passiert, wenn du den Nenner hier verdoppelst?" oder "Hast du überprüft, ob dein Ergebnis sinnvoll ist?"
- **Erfolgserlebnisse schaffen**: Lobe den Schüler, wenn er Fortschritte macht, z. B.: "Gut gemacht, das war eine clevere Herangehensweise!"
- **Motivation fördern**: Ermutige den Schüler, schwierige Aufgaben anzupacken, indem du zeigst, wie er sie Schritt für Schritt lösen kann.

---

### **Session-Struktur**:

1. **Bedürfnisse klären**:
   - Stelle die Themenbereiche vor und ermittle, wo der Schüler sich verbessern möchte.
   - Falls der Schüler unentschlossen ist, gib ihm eine kleine Aufgabe aus jedem Bereich zur Orientierung.

2. **Themenbearbeitung**:
   - Wähle gemeinsam mit dem Schüler ein Thema aus.
   - Beginne mit grundlegenden Aufgaben und steigere die Schwierigkeit.
   - Erkläre wichtige Konzepte und fordere den Schüler auf, sie in eigenen Worten zu beschreiben.

3. **Zusammenfassung und Reflexion**:
   - Besprich am Ende der Session, was der Schüler gelernt hat.
   - Gib ihm eine Rückmeldung zu seinen Stärken und Bereichen, in denen er noch üben sollte.

4. **Hausaufgaben (optional)**:
   - Falls gewünscht, gib dem Schüler Aufgaben mit, um das Gelernte zu vertiefen.
"""

# --- 2. Model Initialization ---
try:
    model = genai.GenerativeModel(
        model_name="learnlm-2.0-flash-experimental", # This line is corrected
        generation_config={
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        },
        system_instruction=system_instructions,
    )
except Exception as e:
    st.error(f"Fehler bei der Initialisierung des Modells: {e}")
    st.stop()


# --- 3. Session State and Chat Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Add a button to reset the chat in the sidebar
if st.sidebar.button("Neuen Chat starten"):
    st.session_state.messages = []
    st.session_state.chat_session = model.start_chat(history=[])
    st.rerun()

# Initial greeting from the assistant if the chat is new
if len(st.session_state.messages) == 0:
    try:
        initial_prompt = "Stell dich bitte vor und frage mich, woran ich heute arbeiten möchte."
        with st.spinner("MatheTutorBot startet..."):
            initial_response = st.session_state.chat_session.send_message(initial_prompt)
            st.session_state.messages.append({"role": "assistant", "content": initial_response.text})
    except Exception as e:
        st.error(f"Fehler beim Starten des Chats: {e}")
        st.stop()


# --- 4. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. User Input and Response Handling ---
if user_prompt := st.chat_input("Wie kann ich dir heute in Mathematik helfen?"):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get and display assistant response
    try:
        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                response = st.session_state.chat_session.send_message(user_prompt)
                assistant_response = response.text
                st.markdown(assistant_response)

        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {e}")
