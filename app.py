import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search, compute_phrase_embeddings, MODEL_CONFIG
import numpy as np
import psutil, os, time
import torch, transformers, sentence_transformers

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

@st.cache_data(show_spinner=False)
def get_data():
    df = load_all_excels()
    df = compute_phrase_embeddings(df)
    return df

df = get_data()

# 📊 Блок отладки в sidebar
with st.sidebar:
    st.markdown("### ⚙️ Отладка")

    st.write("Модель:", MODEL_CONFIG["name"])
    st.write("Префиксы:", MODEL_CONFIG["add_prefix"])

    st.write("Фраз загружено:", len(df))
    st.write("Кол-во тем:", len({t for ts in df['topics'] for t in ts}))
    st.write("Размерность эмбеддингов:", df.attrs.get("emb_dim", "-"))

    avg_len = int(df["phrase"].str.len().mean())
    max_len = int(df["phrase"].str.len().max())
    st.write(f"Средняя длина фразы: {avg_len} симв.")
    st.write(f"Макс. длина фразы: {max_len} симв.")

    if "embedding_time" in df.attrs:
        st.write("Время пересчёта эмбеддингов:", round(df.attrs["embedding_time"], 3), "сек.")

    if "last_query_time" in st.session_state:
        st.write("Время ответа:", round(st.session_state["last_query_time"], 3), "сек.")

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    st.write(f"CPU: {cpu}%")
    st.write(f"RAM: {ram.percent}% ({mem_info.rss/1024**2:.1f} MB)")

    st.write("Torch:", torch.__version__)
    st.write("Transformers:", transformers.__version__)
    st.write("Sentence-Transformers:", sentence_transformers.__version__)

# 🔘 Фильтр по тематикам
all_topics = sorted({topic for topics in df['topics'] for topic in topics})
selected_topics = st.multiselect("Фильтр по тематикам (независимо от поиска):", all_topics)
filter_search_by_topics = st.checkbox("Искать только в выбранных тематиках", value=False)

if selected_topics:
    st.markdown("### 📂 Фразы по выбранным тематикам:")
    filtered_df = df[df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))]
    for row in filtered_df.itertuples():
        with st.container():
            st.markdown(f"""
                <div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                    <div style="font-size: 18px; font-weight: 600; color: #333;">📝 {row.phrase_full}</div>
                    <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(row.topics)}</strong></div>
                </div>""", unsafe_allow_html=True)
            if row.comment and str(row.comment).strip().lower() != "nan":
                with st.expander("💬 Комментарий", expanded=False):
                    st.markdown(row.comment)

# 📥 Поисковый запрос
query = st.text_input("Введите ваш запрос:")

if query:
    try:
        search_df = df
        if filter_search_by_topics and selected_topics:
            mask = df['topics'].apply(lambda topics: any(t in selected_topics for t in topics))
            search_df = df[mask].reset_index(drop=True)

            full_embs = df.attrs.get('phrase_embs', None)
            full_norms = df.attrs.get('phrase_embs_norms', None)
            if full_embs is not None and full_norms is not None:
                indices = list(np.where(mask.values)[0])
                if indices:
                    search_df.attrs['phrase_embs'] = full_embs[indices]
                    search_df.attrs['phrase_embs_norms'] = full_norms[indices]
                else:
                    emb_dim = full_embs.shape[1] if full_embs.size else 0
                    search_df.attrs['phrase_embs'] = np.zeros((0, emb_dim), dtype='float32')
                    search_df.attrs['phrase_embs_norms'] = np.zeros((0,), dtype='float32')

        if search_df.empty:
            st.warning("Нет данных для поиска по выбранным тематикам.")
        else:
            start = time.time()
            results = semantic_search(query, search_df)
            st.session_state["last_query_time"] = time.time() - start

            if results:
                st.markdown("### 🔍 Результаты умного поиска:")
                for score, phrase_full, topics, comment in results:
                    with st.container():
                        st.markdown(f"""
                            <div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                <div style="font-size: 18px; font-weight: 600; color: #333;">🧠 {phrase_full}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                                <div style="margin-top: 2px; font-size: 13px; color: #999;">🎯 Релевантность: {score:.2f}</div>
                            </div>""", unsafe_allow_html=True)
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.warning("Совпадений не найдено в умном поиске.")

            exact_results = keyword_search(query, search_df)
            if exact_results:
                st.markdown("### 🧷 Точный поиск:")
                for phrase, topics, comment in exact_results:
                    with st.container():
                        st.markdown(f"""
                            <div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: #f9f9f9; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                                <div style="font-size: 18px; font-weight: 600; color: #333;">📌 {phrase}</div>
                                <div style="margin-top: 4px; font-size: 14px; color: #666;">🔖 Тематики: <strong>{', '.join(topics)}</strong></div>
                            </div>""", unsafe_allow_html=True)
                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий", expanded=False):
                                st.markdown(comment)
            else:
                st.info("Ничего не найдено в точном поиске.")
    except Exception as e:
        st.error(f"Ошибка при обработке запроса: {e}")
