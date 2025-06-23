import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib
import tempfile
from pathlib import Path
from PIL import Image
import re

matplotlib.use("Agg")

# Define sins and behaviors
sins = ["Lust", "Gluttony", "Greed", "Sloth", "Wrath", "Envy", "Pride"]

behaviors = [
    ("Stealing office snacks", [0, 0.8, 0.7, 0.3, 0.1, 0.6, 0.4]),
    ("Doomscrolling in bed", [0.2, 0.1, 0.2, 0.9, 0.1, 0.3, 0.2]),
    ("Yelling at your roommate", [0, 0.1, 0.1, 0.2, 0.9, 0.3, 0.5]),
    ("Bragging about your crypto gains", [0.1, 0.2, 0.9, 0.1, 0.2, 0.4, 0.95]),
    ("Binge-eating cake while watching reality TV", [0.1, 0.95, 0.3, 0.85, 0.2, 0.2, 0.3]),
]

# Create DataFrame
df_sins = pd.DataFrame([{"Behavior": b[0], **dict(zip(sins, b[1]))} for b in behaviors])

# --- Simple Sin Coding Function ---
def sincode(description):
    desc = description.lower()
    vec = np.zeros(7)
    if any(word in desc for word in ["sex", "onlyfans", "flirting"]):
        vec[0] += 0.9  # Lust
    if any(word in desc for word in ["food", "cake", "eating", "snack"]):
        vec[1] += 0.8  # Gluttony
    if any(word in desc for word in ["money", "crypto", "steal"]):
        vec[2] += 0.9  # Greed
    if any(word in desc for word in ["lazy", "sleeping", "netflix", "doomscroll"]):
        vec[3] += 1.0  # Sloth
    if any(word in desc for word in ["yell", "shout", "fight"]):
        vec[4] += 0.85  # Wrath
    if any(word in desc for word in ["jealous", "envy", "stalking"]):
        vec[5] += 0.75  # Envy
    if any(word in desc for word in ["brag", "flex", "pride"]):
        vec[6] += 0.9  # Pride
    return vec

# --- Optional: LLM Sin Coding Function ---
import json
try:
    import openai
    openai.api_key = st.secrets["openai_api_key"]

    def llm_sincode(sentence):
        try:
            client = openai.OpenAI()
            prompt = f"""
    Return a valid JSON dictionary with 7 numeric values between 0.0 and 1.0. These represent the intensity of the seven cardinal sins in the behavior described below.

    Respond with **only** the JSON object — no explanation, no formatting, no markdown.

    Sentence: "{sentence}"
    """

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a moral classifier. You evaluate behavior according to the 7 cardinal sins. Your job is to output a strict JSON object with keys: Lust, Gluttony, Greed, Sloth, Wrath, Envy, Pride. Return nothing else."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )

            content = response.choices[0].message.content.strip()
            sin_values = json.loads(content)
            return np.array([sin_values[sin] for sin in sins])

        except Exception as e:
            st.warning(f"LLM failed for sentence: \"{sentence}\"\nError: {e}")
            return sincode(sentence)


except Exception as e:
    def llm_sincode(sentence):
        st.warning("LLM sin-coding not available. Falling back to rule-based method.")
        return sincode(sentence)
except Exception as e:
    def llm_sincode(sentence):
        st.warning("LLM sin-coding not available. Falling back to rule-based method.")
        return sincode(sentence)

# --- PCA Component Labeling ---

def label_principal_components(pca_components, sins):
    labels = []
    raw = []
    for i, component in enumerate(pca_components):
        sorted_indices = np.argsort(component)
        high = [(sins[j], component[j]) for j in sorted_indices[-2:][::-1]]
        low = [(sins[j], component[j]) for j in sorted_indices[:2]]
        label = f"PC{i+1}: +{high[0][0]} / +{high[1][0]} vs -{low[0][0]} / -{low[1][0]}"
        labels.append(label)
        raw.append({"PC": f"PC{i+1}", "top_pos": high, "top_neg": low})
    return labels, raw

# --- LLM-Based PCA Label Description ---
def describe_pca_components(raw_components):
    try:
        client = openai.OpenAI()
        prompt = f"""
Interpret the following PCA loadings derived from sin-based behavior vectors.

Return:
- A short, conceptual label for each PC axis (max 3–4 words each).
- Only output a JSON array like: ["Label for PC1", "Label for PC2"]
- Do NOT include any explanation, formatting, or markdown.

Data:
{json.dumps(raw_components, indent=2)}
"""

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You're a moral-semantic analyst. Your response must be a valid JSON array of two short string labels, no extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        content = response.choices[0].message.content.strip()

        # Try loading directly as JSON
        try:
            return json.loads(content)
        except:
            # Fallback: extract first JSON-like list using regex
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                return ast.literal_eval(match.group(0))

            raise ValueError("Could not extract valid JSON from LLM output.")

    except Exception as e:
        st.warning(f"Failed to generate short PCA labels: {e}")
        return ["PC1", "PC2"]


# --- LLM-Based PCA Label Commentary ---
def comment_pca_components(raw_components):
    try:
        client = openai.OpenAI()
        prompt = f"""Describe the meaning of the following principal component loadings in human-interpretable psychological terms, focusing on their relationships to the seven cardinal sins:

{json.dumps(raw_components, indent=2)}

Return a short interpretation for each principal component."""

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a philosophical psychologist trained in moral cognition."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"Failed to generate PCA interpretation: {e}")
        return None


# --- PCA and Plotting ---

def plot_trajectory_with_labels(df_selected, sins, pc_labels):
    pca = PCA(n_components=2)
    df_selected = df_selected.reset_index(drop=True)
    X_pca = pca.fit_transform(df_selected[sins].values)
    # pc_labels, _ = label_principal_components(pca.components_, sins)
    df_selected[["PC1", "PC2"]] = X_pca

    steps_between = 15
    total_steps = (len(df_selected) - 1) * steps_between
    angles = np.linspace(0, 2 * np.pi, len(sins), endpoint=False).tolist() + [0]

    interp_pca, interp_sins = [], []
    for i in range(len(df_selected) - 1):
        p1 = df_selected.iloc[i][["PC1", "PC2"]].values
        p2 = df_selected.iloc[i + 1][["PC1", "PC2"]].values
        s1 = df_selected.iloc[i][sins].values
        s2 = df_selected.iloc[i + 1][sins].values
        for alpha in np.linspace(0, 1, steps_between):
            interp_pca.append((1 - alpha) * p1 + alpha * p2)
            interp_sins.append((1 - alpha) * s1 + alpha * s2)

    colors = cm.magma(np.linspace(0, 1, total_steps))

    fig = plt.figure(figsize=(12, 6))
    # Set dark background for the entire figure
    fig.patch.set_facecolor('#2E2E2E')

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, polar=True)

    # Set dark theme colors for both subplots
    ax1.set_facecolor('#3C3C3C')
    ax2.set_facecolor('#3C3C3C')

    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Reapply dark theme after clearing
        ax1.set_facecolor('#3C3C3C')
        ax2.set_facecolor('#3C3C3C')
        
        ax1.scatter(df_selected["PC1"], df_selected["PC2"], alpha=0.3, marker='$🔥$')
        for i in range(frame):
            c = colors[i]
            ax1.plot(
                [interp_pca[i][0], interp_pca[i + 1][0]],
                [interp_pca[i][1], interp_pca[i + 1][1]],
                color=c, linewidth=4
            )
        
        # Set title and labels with light colors
        ax1.set_title(f"Interpolated Sin Trajectory in PCA Space", 
                      color='white', fontsize=12)
        ax1.set_xlabel(pc_labels[0], color='white')
        ax1.set_ylabel(pc_labels[1], color='white')
        ax1.grid(True, color='#666666', alpha=0.7)
        
        # Set tick colors to white
        ax1.tick_params(colors='white')
        
        # Set spine colors to light gray
        for spine in ax1.spines.values():
            spine.set_color('#CCCCCC')
        
        values = interp_sins[frame].tolist() + [interp_sins[frame][0]]
        radar_color = colors[frame]
        ax2.plot(angles, values, color=radar_color, linewidth=2)
        ax2.fill(angles, values, color=radar_color, alpha=0.3)
        ax2.set_thetagrids(np.degrees(angles[:-1]), sins)
        ax2.set_title("Interpolated Sin Profile", color='white', fontsize=12)
        
        # Set polar plot colors
        ax2.tick_params(colors='white')
        ax2.grid(True, color='#666666', alpha=0.7)
        
        # Set radial labels color
        ax2.set_rlabel_position(0)
        for label in ax2.get_yticklabels():
            label.set_color('white')

    anim = FuncAnimation(fig, update, frames=total_steps - 1, interval=100)

    temp_dir = tempfile.gettempdir()
    gif_path = Path(temp_dir) / "sin_trajectory.gif"
    anim.save(gif_path, writer="pillow")

    st.image(str(gif_path), caption="Sin Trajectory Animation", use_container_width=True)
    st.download_button("Download GIF", data=open(gif_path, "rb"), file_name="sin_trajectory.gif", mime="image/gif")

    return pc_labels

# Streamlit UI
st.title('Sin Space Trajectory Visualizer')

# Sin coding mode toggle
use_llm = st.checkbox("Use LLM for sin coding (requires API key)")

# Add longer narrative input
text_block = st.text_area("Describe a full narrative or story (each sentence will be sin-coded):", key="text_block")
create_button = st.button("Create Sin Trajectory")
selected_behaviors = []

if create_button and text_block:
    with st.spinner("🔥Processing sin trajectory..."):
        # Naive sentence segmentation
        sentences = re.split(r'(?<=[.!?])\s+', text_block.replace("\n", " ").strip())
        for sent in sentences:
            if sent:
                encoded = llm_sincode(sent) if use_llm else sincode(sent)
                new_entry = {"Behavior": sent.strip(), **dict(zip(sins, encoded))}
                df_sins = pd.concat([df_sins, pd.DataFrame([new_entry])], ignore_index=True)
        selected_behaviors = [s.strip() for s in sentences if s.strip()]

        df_selected = df_sins[df_sins["Behavior"].isin(selected_behaviors)]

        if len(df_selected) >= 2:
            # pc_labels, raw_labels = label_principal_components(PCA(n_components=2).fit(df_selected[sins].values).components_, sins)
            # plot_trajectory_with_labels(df_selected, sins, pc_labels)
            pca_model = PCA(n_components=2).fit(df_selected[sins].values)
            pc_labels, raw_labels = label_principal_components(pca_model.components_, sins)
            semantic_labels = describe_pca_components(raw_labels)
            plot_trajectory_with_labels(df_selected, sins, semantic_labels)


            # Display interpretation below the animation
            # interpretation = describe_pca_components(raw_labels)
            # if interpretation:
            #     st.subheader("🧠 Interpreted PCA Dimensions")
            #     st.markdown(interpretation)
        else:
            st.warning("❗ Need at least two valid sin-coded sentences to compute PCA and visualize.")

    # for sent in sentences:
    #     if sent:
    #         encoded = llm_sincode(sent) if use_llm else sincode(sent)
    #         new_entry = {"Behavior": sent.strip(), **dict(zip(sins, encoded))}
    #         df_sins = pd.concat([df_sins, pd.DataFrame([new_entry])], ignore_index=True)
    # selected_behaviors = [s.strip() for s in sentences if s.strip()]

    # # df_selected = df_sins[df_sins["Behavior"].isin(selected_behaviors)]
    # # pc_labels, raw_labels = label_principal_components(PCA(n_components=2).fit(df_selected[sins].values).components_, sins)
    # # plot_trajectory_with_labels(df_selected, sins, pc_labels)
    # pca_model = PCA(n_components=2).fit(df_selected[sins].values)
    pc_labels, raw_labels = label_principal_components(pca_model.components_, sins)
    # semantic_labels = describe_pca_components(raw_labels)
    # plot_trajectory_with_labels(df_selected, sins, semantic_labels)

    # Display interpretation below the animation
    interpretation = comment_pca_components(raw_labels)
    if interpretation:
        try:
            # # Extract short PC labels and a summary
            lines = interpretation.strip().splitlines()
            # short_labels = []
            # summary_lines = []
            # for line in lines:
            #     if line.lower().startswith("pc"):
            #         label = line.split(":", 1)[1].strip()
            #         short_labels.append(label)
            #     else:
            #         summary_lines.append(line.strip())

            # if len(short_labels) >= 2:
            #     pc_labels = short_labels[:2]  # Use only PC1 and PC2 labels
            # else:
            #     st.warning("Could not extract short PCA labels from LLM.")

            st.subheader("🧠 Interpreted PCA Dimensions")
            st.markdown("\n".join(lines))

        except Exception as e:
            st.warning(f"⚠️ Error parsing LLM PCA interpretation: {e}")

