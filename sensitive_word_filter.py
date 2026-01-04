import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from collections import deque, defaultdict
import math
import pandas as pd

st.set_page_config(page_title="Sensitive Word Filter",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .highlight {
        background-color: #FF4B4B;
        color: white;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .success-box {
        background-color: #1F8B24;
        padding: 12px;
        border-radius: 5px;
        text-align: center;
        color: white;
        font-size: 1rem;
    }
    .centered-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        margin-top: 0.5rem;
    }
    .centered-credit {
        text-align: center;
        font-size: 1rem;
        color: #B0B0B0;
        margin-bottom: 0.3rem;
    }
    .transition-step {
        background-color: #1E3A5F;
        padding: 14px;
        border-radius: 5px;
        margin: 8px 0;
        border-left: 4px solid #4A9EFF;
        font-size: 1.05rem;
        line-height: 1.8;
    }
    .match-found {
        background-color: #2D5016;
        padding: 14px;
        border-radius: 5px;
        margin: 8px 0;
        border-left: 4px solid #2ECC71;
        font-size: 1.05rem;
        line-height: 1.8;
    }
    .result-text {
        line-height: 1.8;
        word-wrap: break-word;
        font-size: 1.1rem;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem !important;
    }
    h2, h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.6rem !important;
    }
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    p {
        margin-bottom: 0.6rem !important;
        font-size: 1rem !important;
    }
    .main .block-container {
        padding-bottom: 5rem !important;
    }
    div[data-testid="stDataFrame"] > div > div > div > div > table {
        font-size: 0.9rem !important;
    }
    div[data-testid="stDataFrame"] > div > div > div > div > table td,
    div[data-testid="stDataFrame"] > div > div > div > div > table th {
        padding: 4px 8px !important;
        line-height: 1.2 !important;
    }
    </style>
    """, unsafe_allow_html=True)


class AhoCorasick:
    def __init__(self):
        self.goto = {}
        self.fail = {}
        self.output = defaultdict(list)
        self.states_count = 1

    def add_word(self, word):
        word = word.lower()
        state = 0
        for char in word:
            if (state, char) not in self.goto:
                self.goto[(state, char)] = self.states_count
                self.states_count += 1
            state = self.goto[(state, char)]
        self.output[state].append(word)

    def build(self):
        queue = deque()

        for char in set(char for (state, char) in self.goto.keys() if state == 0):
            state = self.goto[(0, char)]
            self.fail[state] = 0
            queue.append(state)

        while queue:
            current_state = queue.popleft()

            for (state, char), next_state in self.goto.items():
                if state == current_state:
                    queue.append(next_state)

                    fail_state = self.fail.get(current_state, 0)
                    while fail_state != 0 and (fail_state, char) not in self.goto:
                        fail_state = self.fail.get(fail_state, 0)

                    if (fail_state, char) in self.goto and self.goto[(fail_state, char)] != next_state:
                        self.fail[next_state] = self.goto[(fail_state, char)]
                    else:
                        self.fail[next_state] = 0

                    if self.fail[next_state] in self.output:
                        self.output[next_state].extend(
                            self.output[self.fail[next_state]])

    def search_with_path(self, text):
        original_text = text
        text_lower = text.lower()
        state = 0
        results = []
        path = []

        for i, char in enumerate(text_lower):
            transitions = []

            while state != 0 and (state, char) not in self.goto:
                fail_state = self.fail.get(state, 0)
                transitions.append({
                    'type': 'fail',
                    'from': state,
                    'to': fail_state,
                    'char': char,
                    'position': i
                })
                state = fail_state

            if (state, char) in self.goto:
                next_state = self.goto[(state, char)]
                transitions.append({
                    'type': 'goto',
                    'from': state,
                    'to': next_state,
                    'char': char,
                    'position': i
                })
                state = next_state

            matches = []
            if state in self.output:
                for word in self.output[state]:
                    start_pos = i - len(word) + 1
                    end_pos = i
                    matched_text = original_text[start_pos:end_pos + 1]

                    results.append((start_pos, word))
                    matches.append({
                        'word': word,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'matched_text': matched_text
                    })

            path.append({
                'position': i,
                'char': char,
                'transitions': transitions,
                'final_state': state,
                'matches': matches
            })

        return results, path

    def visualize(self):
        G = nx.DiGraph()

        for state in range(self.states_count):
            label = f"S{state}"
            if state in self.output:
                words = ','.join(self.output[state])
                label += f"\n{words}"
            G.add_node(state, label=label)

        for (state, char), next_state in self.goto.items():
            G.add_edge(state, next_state, label=char, edge_type='goto')

        for state, fail_state in self.fail.items():
            if fail_state != 0:
                G.add_edge(state, fail_state, edge_type='fail')

        return G

    def get_complexity_metrics(self):
        return {
            'num_states': self.states_count,
            'num_goto_edges': len(self.goto),
            'num_fail_edges': len(self.fail),
            'num_output_states': len(self.output)
        }

    def get_transition_table(self):
        all_chars = sorted(set(char for (state, char) in self.goto.keys()))

        table_data = []
        for state in range(self.states_count):
            row = {'State': f'S{state}'}

            for char in all_chars:
                if (state, char) in self.goto:
                    row[f'δ({char})'] = f'S{self.goto[(state, char)]}'
                else:
                    row[f'δ({char})'] = '-'

            if state in self.fail:
                fail_to = self.fail[state]
                row['Fail Link'] = f'S{fail_to}'
            else:
                row['Fail Link'] = '-'

            if state in self.output:
                row['Output'] = ', '.join(self.output[state])
            else:
                row['Output'] = '-'

            table_data.append(row)

        return pd.DataFrame(table_data)


def get_hierarchical_layout(G, automaton):
    levels = {0: 0}
    queue = deque([0])

    while queue:
        state = queue.popleft()
        current_level = levels[state]

        for (s, char), next_state in automaton.goto.items():
            if s == state and next_state not in levels:
                levels[next_state] = current_level + 1
                queue.append(next_state)

    nodes_by_level = defaultdict(list)
    for node, level in levels.items():
        nodes_by_level[level].append(node)

    pos = {}
    max_level = max(levels.values()) if levels else 0

    for level, nodes in nodes_by_level.items():
        num_nodes = len(nodes)
        y = -level * 250.0

        if num_nodes == 1:
            x_positions = [0]
        else:
            total_width = max(20.0, num_nodes * 3.5)
            x_positions = [i * (total_width / (num_nodes - 1)) - total_width / 2
                           for i in range(num_nodes)]

        nodes_sorted = sorted(nodes)
        for i, node in enumerate(nodes_sorted):
            pos[node] = (x_positions[i], y)

    return pos, max_level


def get_best_layout(G, automaton, num_nodes):
    try:
        pos, max_level = get_hierarchical_layout(G, automaton)
        if pos and len(pos) == num_nodes:
            return pos, max_level
    except Exception as e:
        print(f"Hierarchical layout failed: {e}")

    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
        return pos, 5
    except:
        pass

    try:
        if num_nodes <= 25:
            pos = nx.kamada_kawai_layout(G, scale=5.0)
            return pos, 5
    except:
        pass

    pos = nx.spring_layout(G, k=15/math.sqrt(num_nodes),
                           iterations=200, seed=42, scale=4.5)
    return pos, 5


def create_interactive_graph(G, active_states, active_goto_edges, active_fail_edges, automaton, num_nodes, show_fail_edges=False):
    pos, max_level = get_best_layout(G, automaton, num_nodes)

    if active_states:
        active_states = active_states | {0}

    edge_traces = []
    annotations = []

    if num_nodes < 10:
        node_size = 70
        font_size = 12
    elif num_nodes < 20:
        node_size = 65
        font_size = 11
    else:
        node_size = 60
        font_size = 10

    for (u, v, d) in G.edges(data=True):
        if d.get('edge_type') != 'goto':
            continue

        is_active = (u, v) in active_goto_edges

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=4 if is_active else 2,
                      color='#FF8C00' if is_active else '#4A9EFF'),
            opacity=1.0 if is_active else 0.35,
            hoverinfo='none',
            showlegend=False
        ))

        dx = x1 - x0
        dy = y1 - y0

        arrow_start_x = x0 + dx * 0.68
        arrow_start_y = y0 + dy * 0.68
        arrow_end_x = x0 + dx * 0.72
        arrow_end_y = y0 + dy * 0.72

        annotations.append(dict(
            x=arrow_end_x,
            y=arrow_end_y,
            ax=arrow_start_x,
            ay=arrow_start_y,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=2.0,
            arrowcolor='#FF8C00' if is_active else '#4A9EFF',
            opacity=1.0 if is_active else 0.35
        ))

    if show_fail_edges:
        for (u, v, d) in G.edges(data=True):
            if d.get('edge_type') != 'fail':
                continue

            is_active = (u, v) in active_fail_edges

            x0, y0 = pos[u]
            x1, y1 = pos[v]

            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=3.5 if is_active else 2,
                          color='#FF1744' if is_active else '#FF5252', dash='dash'),
                opacity=1.0 if is_active else 0.6,
                hoverinfo='none',
                showlegend=False
            ))

            dx = x1 - x0
            dy = y1 - y0

            arrow_start_x = x0 + dx * 0.68
            arrow_start_y = y0 + dy * 0.68
            arrow_end_x = x0 + dx * 0.72
            arrow_end_y = y0 + dy * 0.72

            annotations.append(dict(
                x=arrow_end_x,
                y=arrow_end_y,
                ax=arrow_start_x,
                ay=arrow_start_y,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowwidth=2.0,
                arrowcolor='#FF1744' if is_active else '#FF5252',
                opacity=1.0 if is_active else 0.6
            ))

    edge_label_traces = []
    for u, v, d in G.edges(data=True):
        if d.get('edge_type') == 'goto':
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            label_x = (x0 + x1) / 2
            label_y = (y0 + y1) / 2

            is_active_edge = (u, v) in active_goto_edges

            edge_label_traces.append(go.Scatter(
                x=[label_x],
                y=[label_y],
                mode='text',
                text=[d.get('label', '')],
                textfont=dict(
                    size=16 if is_active_edge else 14,
                    color='#FFD700' if is_active_edge else '#FFFFFF',
                    family='Arial Black'
                ),
                textposition='middle center',
                hoverinfo='none',
                showlegend=False
            ))

    node_x = []
    node_y = []
    node_colors = []
    node_labels = []
    node_hover = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if node in active_states:
            node_colors.append('#FF8C00')
        elif node in automaton.output:
            node_colors.append('#2ECC71')
        else:
            node_colors.append('#34495E')

        label = G.nodes[node]['label']
        node_labels.append(label)

        state_info = []
        if node == 0:
            state_info.append("Start")
        if node in automaton.output:
            state_info.append("Match")

        hover_text = f"State: {node}"
        if state_info:
            hover_text += f" ({', '.join(state_info)})"
        hover_text += f"<br>Label: {label.replace(chr(92)+'n', ', ')}"
        node_hover.append(hover_text)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color=node_colors,
            line=dict(width=3, color='#ECF0F1')
        ),
        text=node_labels,
        textfont=dict(size=font_size, color='white', family='Arial Black'),
        textposition='middle center',
        hovertext=node_hover,
        hoverinfo='text',
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + edge_label_traces + [node_trace])
    fig.update_layout(annotations=annotations)

    y_min = -max_level * 250.0 - 50
    y_max = 50

    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=5, l=5, r=5, t=5),
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[
                   y_min, y_max], fixedrange=False, scaleanchor=None),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        height=1000,
        dragmode='pan'
    )

    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['resetScale2d'],
        'modeBarButtonsToRemove': ['select2d', 'lasso2d']
    }

    return fig, config


if 'sensitive_words' not in st.session_state:
    st.session_state.sensitive_words = [
        'kill', 'bomb', 'suicide', 'weapon', 'explosive']
if 'automaton' not in st.session_state:
    st.session_state.automaton = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'user_message' not in st.session_state:
    st.session_state.user_message = ""
if 'transition_path' not in st.session_state:
    st.session_state.transition_path = []


def build_automaton():
    ac = AhoCorasick()
    for word in st.session_state.sensitive_words:
        ac.add_word(word)
    ac.build()
    st.session_state.automaton = ac


if st.session_state.automaton is None:
    build_automaton()

st.markdown('<div class="centered-title">🖥️ Sensitive Word Filter by Aho-Corasick Automaton</div>',
            unsafe_allow_html=True)
st.markdown('<div class="centered-credit">👨‍💻 Developed by Andy Ting Zhi Wei</div>',
            unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Sensitive Word List")

    if st.session_state.sensitive_words:
        st.write("Current words:")
        for idx, word in enumerate(st.session_state.sensitive_words):
            col_word, col_delete = st.columns([3, 1])
            with col_word:
                st.write(f"{idx + 1}. {word}")
            with col_delete:
                if st.button("🗑️", key=f"delete_{word}_{idx}", help=f"Delete '{word}'"):
                    st.session_state.sensitive_words.remove(word)
                    build_automaton()
                    if st.session_state.user_message:
                        results, path = st.session_state.automaton.search_with_path(
                            st.session_state.user_message)
                        st.session_state.search_results = results
                        st.session_state.transition_path = path
                    st.success(f"Deleted '{word}'!")
                    st.rerun()
    else:
        st.info("No sensitive words. Please add some below!")

    st.markdown("---")

    st.subheader("➕ Add New Word")
    new_word = st.text_input("Enter sensitive word:", key="new_word_input")

    if st.button("Add Word"):
        if new_word and new_word.strip():
            new_word_clean = new_word.strip().lower()
            if new_word_clean not in st.session_state.sensitive_words:
                st.session_state.sensitive_words.append(new_word_clean)
                build_automaton()
                if st.session_state.user_message:
                    results, path = st.session_state.automaton.search_with_path(
                        st.session_state.user_message)
                    st.session_state.search_results = results
                    st.session_state.transition_path = path
                st.success(f"Added '{new_word_clean}'!")
                st.rerun()
            else:
                st.warning("The word already exists!")
        else:
            st.error("Please enter a word!")

    st.markdown("---")

    st.subheader("💬 Check Your Message")
    user_input = st.text_area("Enter your message:",
                              height=100, key="message_input")

    if st.button("Check Message"):
        if user_input:
            st.session_state.user_message = user_input
            results, path = st.session_state.automaton.search_with_path(
                user_input)
            st.session_state.search_results = results
            st.session_state.transition_path = path
            st.rerun()
        else:
            st.warning("Please enter a message!")

with col2:
    st.subheader("💻 Automaton Visualization")

    if st.session_state.automaton and st.session_state.sensitive_words:
        G = st.session_state.automaton.visualize()

        metrics = st.session_state.automaton.get_complexity_metrics()
        num_nodes = metrics['num_states']

        col_metric, col_toggle = st.columns([2, 1])
        with col_metric:
            st.caption(
                f"📊 {num_nodes} states, {metrics['num_goto_edges']} transitions, {metrics['num_fail_edges']} fail transitions")
        with col_toggle:
            show_fail = st.checkbox(
                "Show fail transitions", value=False, key="show_fail_edges")

        tab1, tab2 = st.tabs(["📊 Graph View", "📋 Transition Table"])

        with tab1:
            active_states = set()
            active_goto_edges = set()
            active_fail_edges = set()

            if st.session_state.transition_path:
                for step in st.session_state.transition_path:
                    active_states.add(step['final_state'])
                    for trans in step['transitions']:
                        if trans['type'] == 'goto':
                            active_goto_edges.add((trans['from'], trans['to']))
                        else:
                            active_fail_edges.add((trans['from'], trans['to']))

            fig, config = create_interactive_graph(
                G, active_states, active_goto_edges, active_fail_edges,
                st.session_state.automaton, num_nodes, show_fail_edges=show_fail
            )

            st.plotly_chart(fig, width='stretch', config=config)
            st.caption(
                "🟠 Active States | 🟢 Match States | 🔵 Transitions | 🔴 Fail Transitions")

        with tab2:
            df = st.session_state.automaton.get_transition_table()

            st.dataframe(
                df,
                width='stretch',
                hide_index=True
            )
            st.caption(
                "δ(char) = Transition function | Fail Link = Failure transition (fallback path) | Output = Matched words")

    elif not st.session_state.sensitive_words:
        st.info("📭 Please add some sensitive words to see the automaton!")

    if st.session_state.transition_path:
        st.markdown("---")
        st.subheader("📊 Transition Details")

        with st.expander("🔄 View Step-by-Step Transitions", expanded=False):
            for step in st.session_state.transition_path:
                char_display = step['char']
                position = step['position']

                for trans in step['transitions']:
                    if trans['type'] == 'goto':
                        st.markdown(f'<div class="transition-step">📍 Position {position}: Read "{char_display}" → Goto S{trans["from"]} → S{trans["to"]}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="transition-step">↩️ Position {position}: Failed at S{trans["from"]}, Fall back to S{trans["to"]}</div>',
                                    unsafe_allow_html=True)

                if step['matches']:
                    for match in step['matches']:
                        st.markdown(f'<div class="match-found">✅ Match: "{match["word"]}" at pos {match["start_pos"]} → "{match["matched_text"]}"</div>',
                                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 Detection Results")

    if st.session_state.user_message:
        if st.session_state.search_results:
            st.write("**Sensitive words detected:**")

            highlighted_text = st.session_state.user_message
            results_sorted = sorted(
                st.session_state.search_results, key=lambda x: x[0], reverse=True)

            for pos, word in results_sorted:
                original_word = st.session_state.user_message[pos:pos+len(
                    word)]
                highlighted_text = (
                    highlighted_text[:pos] +
                    f'<span class="highlight">{original_word}</span>' +
                    highlighted_text[pos+len(word):]
                )

            st.markdown(f'<div class="result-text" style="background-color: #262730; padding: 15px; border-radius: 5px; border-left: 4px solid #FF4B4B;">{highlighted_text}</div>',
                        unsafe_allow_html=True)

            st.write("**Details:**")
            for pos, word in st.session_state.search_results:
                matched_text = st.session_state.user_message[pos:pos+len(word)]
                st.write(
                    f"• Found '{word}' at position {pos} → '{matched_text}'")
        else:
            st.markdown('<div class="success-box">✅ No sensitive words found! Your message is safe.</div>',
                        unsafe_allow_html=True)
    else:
        st.info("Enter a message and click 'Check Message' to see results.")
