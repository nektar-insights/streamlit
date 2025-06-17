import streamlit as st
import time
from datetime import datetime, timedelta
import threading

# Custom CSS for styling
st.markdown("""
<style>
    .timer-container {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 2px solid #374151;
    }
    
    .timer-display {
        font-family: 'Courier New', monospace;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    .timer-green {
        background-color: #10b981;
    }
    
    .timer-yellow {
        background-color: #f59e0b;
    }
    
    .timer-red {
        background-color: #ef4444;
    }
    
    .timer-flash {
        background-color: #dc2626;
        animation: flash 1s infinite;
    }
    
    @keyframes flash {
        0%, 50% { background-color: #dc2626; }
        51%, 100% { background-color: #b91c1c; }
    }
    
    .add-timer-section {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border: 2px solid #374151;
    }
    
    .stButton > button {
        width: 100%;
    }
    
    .timer-controls {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'timers' not in st.session_state:
        st.session_state.timers = [
            {
                'id': 1,
                'name': 'Topic 1',
                'duration_minutes': 5,
                'time_left': 300,  # 5 minutes in seconds
                'is_running': False,
                'start_time': None,
                'pause_time': None,
                'overtime': 0,
                'is_overtime': False
            }
        ]
    
    if 'next_id' not in st.session_state:
        st.session_state.next_id = 2
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

def format_time(seconds):
    """Format time in MM:SS format"""
    mins = abs(seconds) // 60
    secs = abs(seconds) % 60
    sign = "+" if seconds < 0 else ""
    return f"{sign}{mins:02d}:{secs:02d}"

def get_timer_color_class(time_left, is_overtime):
    """Get CSS class for timer color based on time left"""
    if is_overtime:
        return "timer-flash"
    elif time_left <= 30:
        return "timer-flash"
    elif time_left <= 60:
        return "timer-red"
    elif time_left <= 180:
        return "timer-yellow"
    else:
        return "timer-green"

def update_timer_state(timer_id):
    """Update timer state based on current time"""
    for timer in st.session_state.timers:
        if timer['id'] == timer_id and timer['is_running']:
            if timer['start_time']:
                elapsed = time.time() - timer['start_time']
                if timer['pause_time']:
                    elapsed -= timer['pause_time']
                
                remaining = timer['duration_minutes'] * 60 - elapsed
                
                if remaining <= 0:
                    timer['is_overtime'] = True
                    timer['overtime'] = abs(remaining)
                    timer['time_left'] = 0
                else:
                    timer['time_left'] = int(remaining)
                    timer['is_overtime'] = False
                    timer['overtime'] = 0

def start_timer(timer_id):
    """Start a timer"""
    for timer in st.session_state.timers:
        if timer['id'] == timer_id:
            if not timer['is_running']:
                timer['is_running'] = True
                timer['start_time'] = time.time()
                timer['pause_time'] = 0
                st.session_state.auto_refresh = True

def pause_timer(timer_id):
    """Pause a timer"""
    for timer in st.session_state.timers:
        if timer['id'] == timer_id:
            if timer['is_running']:
                timer['is_running'] = False
                if not timer['pause_time']:
                    timer['pause_time'] = 0
                timer['pause_time'] += time.time() - timer['start_time']

def reset_timer(timer_id):
    """Reset a timer"""
    for timer in st.session_state.timers:
        if timer['id'] == timer_id:
            timer['is_running'] = False
            timer['time_left'] = timer['duration_minutes'] * 60
            timer['start_time'] = None
            timer['pause_time'] = None
            timer['overtime'] = 0
            timer['is_overtime'] = False

def delete_timer(timer_id):
    """Delete a timer"""
    st.session_state.timers = [t for t in st.session_state.timers if t['id'] != timer_id]

def add_timer(name, duration):
    """Add a new timer"""
    new_timer = {
        'id': st.session_state.next_id,
        'name': name,
        'duration_minutes': duration,
        'time_left': duration * 60,
        'is_running': False,
        'start_time': None,
        'pause_time': None,
        'overtime': 0,
        'is_overtime': False
    }
    st.session_state.timers.append(new_timer)
    st.session_state.next_id += 1

def render_timer(timer):
    """Render a single timer"""
    timer_id = timer['id']
    
    # Update timer state if running
    if timer['is_running']:
        update_timer_state(timer_id)
    
    with st.container():
        st.markdown('<div class="timer-container">', unsafe_allow_html=True)
        
        # Timer header with name and delete button
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Editable timer name
            new_name = st.text_input(
                f"Timer Name",
                value=timer['name'],
                key=f"name_{timer_id}",
                label_visibility="collapsed"
            )
            if new_name != timer['name']:
                timer['name'] = new_name
        
        with col2:
            if st.button("🗑️", key=f"delete_{timer_id}", help="Delete timer"):
                delete_timer(timer_id)
                st.rerun()
        
        # Timer duration setting
        col1, col2 = st.columns([3, 1])
        with col1:
            new_duration = st.number_input(
                "Duration (minutes)",
                min_value=1,
                max_value=999,
                value=timer['duration_minutes'],
                key=f"duration_{timer_id}",
                label_visibility="collapsed"
            )
            if new_duration != timer['duration_minutes']:
                timer['duration_minutes'] = new_duration
                if not timer['is_running']:
                    timer['time_left'] = new_duration * 60
        
        with col2:
            st.write(f"{timer['duration_minutes']} min")
        
        # Timer display
        if timer['is_overtime']:
            display_time = format_time(-timer['overtime'])
            color_class = get_timer_color_class(0, True)
        else:
            display_time = format_time(timer['time_left'])
            color_class = get_timer_color_class(timer['time_left'], False)
        
        st.markdown(
            f'<div class="timer-display {color_class}">{display_time}</div>',
            unsafe_allow_html=True
        )
        
        # Timer controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not timer['is_running']:
                if st.button("▶️ Start", key=f"start_{timer_id}", use_container_width=True):
                    start_timer(timer_id)
                    st.rerun()
            else:
                if st.button("⏸️ Pause", key=f"pause_{timer_id}", use_container_width=True):
                    pause_timer(timer_id)
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset", key=f"reset_{timer_id}", use_container_width=True):
                reset_timer(timer_id)
                st.rerun()
        
        with col3:
            # Status indicator
            if timer['is_running']:
                st.success("Running")
            elif timer['is_overtime']:
                st.error("Overtime!")
            else:
                st.info("Stopped")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_add_timer_section():
    """Render the add new timer section"""
    st.markdown('<div class="add-timer-section">', unsafe_allow_html=True)
    st.subheader("Add New Timer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_timer_name = st.text_input(
            "Timer Name",
            value="New Timer",
            key="new_timer_name"
        )
    
    with col2:
        if st.button("➕ Add Timer", use_container_width=True):
            duration = st.session_state.get('new_timer_duration', 5)
            custom_duration = st.session_state.get('custom_duration', '')
            
            if custom_duration:
                try:
                    duration = int(custom_duration)
                except ValueError:
                    duration = 5
            
            add_timer(new_timer_name, duration)
            st.rerun()
    
    # Duration selection
    st.write("**Duration (minutes):**")
    
    # Preset buttons
    cols = st.columns(6)
    preset_times = [5, 10, 15, 20, 25, 30]
    
    for i, minutes in enumerate(preset_times):
        with cols[i]:
            if st.button(f"{minutes}", key=f"preset_{minutes}", use_container_width=True):
                st.session_state.new_timer_duration = minutes
                st.session_state.custom_duration = ''
    
    # Custom duration input
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_duration = st.number_input(
            "Custom duration",
            min_value=1,
            max_value=999,
            value=None,
            placeholder="Custom minutes",
            key="custom_duration",
            label_visibility="collapsed"
        )
    with col2:
        st.write("minutes")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("⏱️ PTI-Style Meeting Timers")
    
    # Auto-refresh for running timers
    any_running = any(timer['is_running'] for timer in st.session_state.timers)
    if any_running:
        time.sleep(1)
        st.rerun()
    
    # Add new timer section
    render_add_timer_section()
    
    # Render all timers
    if st.session_state.timers:
        for timer in st.session_state.timers:
            render_timer(timer)
    else:
        st.info("No timers created yet. Add a timer above to get started!")
    
    # Auto-refresh toggle
    st.sidebar.header("Settings")
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh (keeps timers updated)",
        value=any_running,
        help="Automatically refresh the page to update running timers"
    )
    
    if auto_refresh and any_running:
        time.sleep(0.5)
        st.rerun()

if __name__ == "__main__":
    main()
