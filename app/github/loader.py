"""GitHub repository loader."""

import base64
import requests


GITHUB_API_BASE = "https://api.github.com"


class GitHubLoader:
    """Loader for GitHub repositories."""

    def __init__(self, token: str | None = None):
        self.session = requests.Session()
        if token:
            self.session.headers.update(
                {"Authorization": f"Bearer {token}"}
            )

    def load_user_repos(self, username: str) -> list[dict[str, any]]:
        """Load repositories for a given GitHub username."""

        repos_url = f"{GITHUB_API_BASE}/users/{username}/repos"
        repos = []

        response = self.session.get(repos_url)
        response.raise_for_status()

        for repo in response.json():
            readme = self._load_readme(username, repo["name"])

            content_parts = [
                f"Repository: {repo['name']}",
                f"Description: {repo.get('description') or 'No description'}",
                f"Primary language: {repo.get('language')}",
            ]

            if readme:
                content_parts.append("README:\n" + readme)

            repos.append(
                {
                    "repo": repo["name"],
                    "content": "\n\n".join(content_parts),
                    "metadata": {
                        "repo": repo["name"],
                        "language": repo.get("language"),
                        "stars": repo.get("stargazers_count"),
                        "url": repo.get("html_url"),
                    },
                }
            )

        return repos

    def _load_readme(self, username: str, repo_name: str) -> str | None:
        """Load the README file for a given repository."""

        url = f"{GITHUB_API_BASE}/repos/{username}/{repo_name}/readme"
        response = self.session.get(url)

        if response.status_code != 200:
            return None

        data = response.json()
        content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
        return content
