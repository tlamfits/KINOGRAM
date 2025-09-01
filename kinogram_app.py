"""
Kinogram App Prototype
======================

This prototype demonstrates the key concepts behind the
kinogram reporting tool described by the user.  The goal
of the application is to allow a coach or analyst to load
a video clip, select a sport‐specific set of positions
(“key frames”) and capture still images from the video
that correspond to those positions.  The captured frames
are combined into a single kinogram report along with
annotations or notes.  Although this script uses Streamlit
for the web UI, the same concepts could be adapted to
other frameworks (e.g. PyQt, Electron, or a native mobile
app).

Key Features:

* Video upload: Users upload a video file for analysis.
* Skill selection: Choose between predefined skill types
  such as “Approach Jump”, “Throwing/Arm Development” and
  “Acceleration”.  Each skill defines its own set of key
  frames.
* Frame capture: While the video is paused, users click a
  button corresponding to a key frame.  The current
  frame is extracted and stored in memory.  Users can
  optionally add notes for each captured frame.
* Report generation: Once all required frames are
  captured, a kinogram report is compiled.  The report is
  rendered as a composite image containing the captured
  frames with captions.  Users can download the report as
  a PNG file.

This script is intentionally kept simple to illustrate the
workflow.  In a production application you would add
error handling, persistent storage, database integration,
and more sophisticated annotation tools (e.g. drawing
lines or angles on top of each frame).  The report
generation could also produce PDF, PowerPoint or other
formats as needed.

Usage:
  $ streamlit run kinogram_app.py

Dependencies:
  streamlit
  opencv-python
  pillow

"""

import io
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import streamlit as st  # type: ignore


@dataclass
class KinogramFrame:
    """Represents a captured kinogram frame and its annotation."""

    label: str
    image: Image.Image
    notes: str = ""


@dataclass
class SkillDefinition:
    """Holds configuration for a particular skill type."""

    name: str
    frames: List[str]
    default_notes: Dict[str, List[str]] = field(default_factory=dict)


def get_skill_definitions() -> Dict[str, SkillDefinition]:
    """Define the available skills and their key frame labels.

    The labels and optional default notes here are drawn from
    the user’s reference material.  See the accompanying
    PDFs for a deeper explanation of each label.
    """
    return {
        "Approach Jump": SkillDefinition(
            name="Approach Jump",
            frames=[
                "Push (1)",
                "Flight Position (2)",
                "Penultimate Step – Initial Contact (2A)",
                "Chair Position (3)",
                "Pigeon Position (4)",
                "Launch (5)",
                "Landing",
            ],
            default_notes={
                "Push (1)": [
                    "Powerful controlled push",  # from PUSH reflection【161684248030298†L56-L65】
                    "Downward slope – slight drop",  # from summary【161684248030298†L56-L63】
                ],
                "Flight Position (2)": [
                    "Hands above shoulders, relaxed",  # from BIG 5【161684248030298†L56-L63】
                    "Trunk upright (<15° flexion)",
                ],
                "Penultimate Step – Initial Contact (2A)": [
                    "Turn foot to 2 o’clock",
                    "Stable and strong contact",
                ],
                "Chair Position (3)": [
                    "Mid‑stance with vertical shin",  # description【161684248030298†L60-L63】
                    "Control downward momentum",
                ],
                "Pigeon Position (4)": [
                    "Lead leg block and hip turn",
                    "Avoid trunk collapse",
                ],
                "Launch (5)": [
                    "Explode after stack position",
                    "Strong arm contribution",
                ],
                "Landing": [
                    "Balanced and absorptive",
                    "Observe center of mass",
                ],
            },
        ),
        "Throwing / Arm Development": SkillDefinition(
            name="Throwing / Arm Development",
            frames=["Coil", "Hammer", "Monster", "Finish", "Land"],
            default_notes={
                "Coil": [
                    "Smooth flow from approach into coil",  # from reflection【303843969583638†L3-L9】
                    "Maintain arm connection",  # from reflection【303843969583638†L5-L9】
                ],
                "Hammer": [
                    "Initiate with lead elbow pull",  # from reflection【303843969583638†L10-L16】
                    "Relaxed arm and violent trunk rotation",  # from reflection【303843969583638†L10-L20】
                ],
                "Monster": [
                    "Full extension at ball contact",  # implied by Monster position【303843969583638†L18-L21】
                    "Lead shoulder dip & violent rotation",  # from Monster【303843969583638†L18-L21】
                ],
                "Finish": [
                    "Hit through the ball",  # from reflection【303843969583638†L22-L24】
                    "Maintain body alignment",  # general cue
                ],
                "Land": [
                    "Soft landing on one or two feet",  # from reflection【303843969583638†L25-L27】
                    "Reset posture for next action",
                ],
            },
        ),
        "Acceleration (Sprint)": SkillDefinition(
            name="Acceleration (Sprint)",
            frames=[
                "Toe‑off",
                "MVP (Max Vertical Projection)",
                "Strike",
                "Touch‑down",
                "Full‑support",
            ],
            default_notes={
                "Toe‑off": [
                    "Rear foot leaves ground – stance leg perpendicular",  # from ALTIS【559691183804756†L341-L350】
                    "Minimal hip extension",  # from toe‑off description【559691183804756†L410-L426】
                ],
                "MVP (Max Vertical Projection)": [
                    "Highest vertical displacement of COM",
                    "Both feet parallel to ground",  # from description of MVP【559691183804756†L341-L346】
                ],
                "Strike": [
                    "Opposite thigh perpendicular to ground",  # from definition【559691183804756†L346-L349】
                    "Front leg knee nearly extended",  # general cue
                ],
                "Touch‑down": [
                    "Swing leg foot contacts ground",  # from definition【559691183804756†L349-L350】
                    "Maintain posture and stiff ankle",
                ],
                "Full‑support": [
                    "Foot directly under pelvis",  # from definition【559691183804756†L349-L351】
                    "Accelerate through mid‑stance",
                ],
            },
        ),
    }


def draw_report(frames: List[KinogramFrame]) -> Image.Image:
    """Create a kinogram montage from a list of captured frames.

    Each frame is resized to a common height and combined
    horizontally.  Captions are drawn underneath each image.
    Notes are drawn below the caption if provided.
    """
    if not frames:
        raise ValueError("No frames captured to build a report.")

    # Determine maximum height for images
    target_height = 240  # pixels
    font = ImageFont.load_default()

    processed: List[Tuple[Image.Image, str, str]] = []
    for frame in frames:
        # Resize image to maintain aspect ratio
        w, h = frame.image.size
        new_w = int(w * (target_height / h))
        resized = frame.image.resize((new_w, target_height))
        processed.append((resized, frame.label, frame.notes))

    # Compute total width and height (including text) of the montage
    text_height = 30  # approximate height for caption and notes
    total_width = sum(img.size[0] for img, _, _ in processed)
    total_height = target_height + 2 * text_height

    montage = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(montage)

    x_offset = 0
    for img, label, notes in processed:
        montage.paste(img, (x_offset, 0))
        # Draw label
        draw.text((x_offset + 5, target_height + 2), label, fill=(0, 0, 0), font=font)
        # Draw notes (wrap text to fit under each image)
        if notes:
            note_lines = notes.split("\n")
            for i, line in enumerate(note_lines):
                draw.text(
                    (x_offset + 5, target_height + 15 + i * 12),
                    line,
                    fill=(50, 50, 50),
                    font=font,
                )
        x_offset += img.size[0]
    return montage


def main() -> None:
    st.set_page_config(page_title="Kinogram App Prototype", layout="wide")
    st.title("Kinogram Report Builder")

    skills = get_skill_definitions()
    skill_names = list(skills.keys())

    # Sidebar configuration
    st.sidebar.header("Configuration")
    selected_skill_name = st.sidebar.selectbox("Select skill", skill_names)
    skill_def = skills[selected_skill_name]

    uploaded_file = st.sidebar.file_uploader(
        "Upload a video (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "m4v"],
    )

    if uploaded_file:
        # Save video to a temporary file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.flush()

        # Initialize video capture
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps else 0

        st.sidebar.markdown(
            f"**Video Info:** {total_frames} frames at {fps:.1f} FPS (duration {duration:.2f} s)"
        )

        # Keep track of captured frames
        if "captured" not in st.session_state:
            st.session_state["captured"] = {}

        # Video display and frame navigation
        st.header("Video")
        current_frame_index = st.slider(
            "Select frame", 0, total_frames - 1, 0, 1, key="frame_slider"
        )

        # Seek to the selected frame and show it
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, channels="RGB", use_column_width=True)
        else:
            st.error("Could not read frame from video.")

        # Frame capture UI
        st.subheader("Capture Kinogram Frames")
        cols = st.columns(len(skill_def.frames))
        for i, label in enumerate(skill_def.frames):
            captured = st.session_state["captured"].get(label)
            button_label = f"Capture {label}" if not captured else f"Captured {label}"
            if cols[i].button(button_label, disabled=captured is not None):
                # Capture current frame and store it
                img_pil = Image.fromarray(rgb_frame)
                default_note_options = skill_def.default_notes.get(label, [])
                st.session_state["captured"][label] = KinogramFrame(
                    label=label,
                    image=img_pil,
                    notes="; ".join(default_note_options),
                )

        # Annotation UI
        if st.session_state["captured"]:
            st.subheader("Annotations")
            for label in skill_def.frames:
                frame_obj: Optional[KinogramFrame] = st.session_state["captured"].get(label)
                if frame_obj:
                    # Provide a text area prefilled with default notes
                    frame_obj.notes = st.text_area(
                        f"Notes for {label}",
                        value=frame_obj.notes,
                        key=f"notes_{label}",
                    )

        # Report generation
        if len(st.session_state["captured"]) == len(skill_def.frames):
            st.success("All frames captured!")
            if st.button("Generate Kinogram Report"):
                # Order frames according to skill definition
                frames_list = [st.session_state["captured"][lbl] for lbl in skill_def.frames]
                report_image = draw_report(frames_list)

                # Display report in the UI
                st.subheader("Kinogram Report")
                st.image(report_image, caption=f"{selected_skill_name} Kinogram Report")

                # Offer download
                buf = io.BytesIO()
                report_image.save(buf, format="PNG")
                st.download_button(
                    label="Download Report as PNG",
                    data=buf.getvalue(),
                    file_name=f"{selected_skill_name.replace(' ', '_').lower()}_kinogram.png",
                    mime="image/png",
                )

    else:
        st.info("Upload a video to begin.")


if __name__ == "__main__":
    main()