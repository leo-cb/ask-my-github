"""Streamlit dashboard rendering GitHub portfolio metrics per user."""

import asyncio

import plotly.express as px
import streamlit as st

from ask_my_github.config import get_settings, require_cloud_llm
from ask_my_github.dashboard.service import (
    extract_technologies,
    format_date,
    get_cloud_llm,
    repo_stats_frame,
    summarize_repo,
    top_by_commits,
    top_by_recent,
)
from ask_my_github.github.ingest import force_reindex, ingest_user

st.set_page_config(page_title="GitHub Portfolio", layout="wide")


@st.cache_resource(show_spinner="Indexing repositories...")
def load_user(username: str):
    """Ingest (or load) a user's vector store and return it once."""
    return asyncio.run(ingest_user(username))


@st.cache_resource(show_spinner=False)
def get_llm():
    """Return the dashboard's shared cloud chat model."""
    return get_cloud_llm()


@st.cache_data(show_spinner="Generating summary...")
def cached_summary(username: str, repo_name: str) -> str:
    """Summarize a repository from its indexed content (cached per repo)."""
    return summarize_repo(load_user(username), get_llm(), repo_name)


@st.cache_data(show_spinner="Analyzing technologies...")
def cached_technologies(username: str) -> dict:
    """Extract languages and libraries for a user (cached)."""
    return extract_technologies(load_user(username), get_llm(), username)


def _reindex_user(username: str) -> None:
    """Delete and rebuild a user's index, then clear cached artifacts."""
    with st.spinner(f"Reindexing repositories for '{username}'..."):
        asyncio.run(force_reindex(username))
    load_user.clear()
    cached_summary.clear()
    cached_technologies.clear()
    st.rerun()


def _usernames() -> list[str]:
    """Return the deduplicated list of usernames from DASHBOARD_USERS."""
    settings = get_settings()
    seen: set[str] = set()
    users: list[str] = []
    for name in (settings.dashboard_users or "").split(","):
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            users.append(name)
    return users


def render_user_tab(username: str) -> None:
    """Render the full dashboard tab for one GitHub user."""
    frame = repo_stats_frame(username)
    if frame.empty:
        st.warning(f"No ingested data found for '{username}'. Run ingestion first.")
        return

    st.subheader(f"@{username}")

    header_left, header_right = st.columns([3, 1])
    with header_right:
        if st.button("Reindex", key=f"reindex_{username}", help="Delete and rebuild the index for this user"):
            _reindex_user(username)

    total_repos = len(frame)
    total_commits = int(frame["author_commit_count"].sum())
    total_stars = int(frame["stars"].sum())
    total_forks = int(frame["forks"].sum())
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Repositories", total_repos)
    metric2.metric("Your commits", total_commits)
    metric3.metric("Stars", total_stars)
    metric4.metric("Forks", total_forks)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**Primary language distribution**")
        languages = frame["language"].astype(str).replace("", "Unknown").fillna("Unknown")
        language_counts = languages.value_counts().reset_index()
        language_counts.columns = ["Language", "Count"]
        st.plotly_chart(
            px.pie(language_counts, names="Language", values="Count"),
            width="stretch",
        )
    with chart_right:
        st.markdown("**Commits by repository**")
        commits = frame[["name", "author_commit_count"]].set_index("name")["author_commit_count"]
        st.bar_chart(commits)

    st.markdown("### Most active repositories")
    for _, row in top_by_commits(frame, 5).iterrows():
        with st.expander(f"{row['name']} — {int(row['author_commit_count'])} commits"):
            st.markdown(
                f"**Language:** {row['language'] or 'unknown'} · "
                f"**Stars:** {int(row['stars'])}"
            )
            if row.get("description"):
                st.markdown(f"> {row['description']}")
            st.markdown(cached_summary(username, row["name"]))
            st.link_button("View on GitHub", row["html_url"])

    st.markdown("### Recently pushed")
    recent = top_by_recent(frame, 5)
    recent_display = recent.copy()
    recent_display["Pushed"] = recent_display["author_pushed_at"].apply(format_date)
    st.dataframe(
        recent_display[
            ["name", "Pushed", "author_commit_count", "language", "stars"]
        ].rename(
            columns={
                "name": "Repository",
                "author_commit_count": "Commits",
                "language": "Language",
                "stars": "Stars",
            }
        ),
        hide_index=True,
    )

    st.markdown("### Technologies")
    technologies = cached_technologies(username)
    if not technologies:
        st.info("No dependency files indexed, so technologies could not be extracted.")
    else:
        for language, libraries in technologies.items():
            st.markdown(f"**{language}**")
            if libraries:
                st.markdown(" · ".join(f"`{library}`" for library in libraries))
            else:
                st.markdown("_no libraries detected_")

    st.markdown("### Repository detail")
    names = frame["name"].tolist()
    selected = st.radio("Select a repository", names, key=f"detail_{username}")
    row = frame[frame["name"] == selected].iloc[0]
    _render_repo_detail(username, row)


def _render_repo_detail(username: str, row) -> None:
    """Render metadata and a generated summary for a single repository."""
    st.markdown(f"#### {row['name']}")
    if row.get("description"):
        st.markdown(row["description"])
    if row.get("is_fork"):
        parent = row.get("parent")
        st.markdown(f"_(fork{f' of {parent}' if parent else ''})_")

    column1, column2, column3 = st.columns(3)
    column1.markdown(f"**Language:** {row.get('language') or 'unknown'}")
    column1.markdown(f"**License:** {row.get('license') or '—'}")
    column1.markdown(f"**Size:** {row.get('size_kb') or 0} KB")
    column2.markdown(f"**Your commits:** {int(row.get('author_commit_count') or 0)}")
    column2.markdown(f"**Stars:** {int(row.get('stars') or 0)}")
    column2.markdown(f"**Forks:** {int(row.get('forks') or 0)}")
    column3.markdown(f"**Created:** {format_date(row.get('created_at'))}")
    column3.markdown(f"**Last pushed:** {format_date(row.get('author_pushed_at'))}")
    column3.markdown(f"**Open issues:** {int(row.get('open_issues') or 0)}")

    if row.get("topics"):
        st.markdown("**Topics:** " + " · ".join(f"`{t}`" for t in row["topics"].split(",")))

    st.markdown(cached_summary(username, row["name"]))
    st.link_button("Open repository", row["html_url"])


def main() -> None:
    """Render the dashboard, one tab per configured user."""
    try:
        require_cloud_llm(get_settings())
    except ValueError as error:
        st.error(str(error))
        return

    users = _usernames()
    if not users:
        st.error(
            "DASHBOARD_USERS is not set. Set it in .env to a comma-separated "
            "list of GitHub usernames to display in the dashboard."
        )
        return

    st.title("GitHub Portfolio")
    if len(users) == 1:
        render_user_tab(users[0])
    else:
        tabs = st.tabs([f"@{user}" for user in users])
        for tab, user in zip(tabs, users):
            with tab:
                render_user_tab(user)


main()
