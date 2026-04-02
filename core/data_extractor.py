"""
Smart Data Extractor
====================
Auto-detect and extract structured data from pages.
Export to JSON, CSV, or formatted text.
"""

import csv
import io
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class DataExtractor:
    """Extract structured data from browser pages and export in various formats."""

    def __init__(self, browser_engine):
        self.browser = browser_engine

    async def extract_all(self, context_id: str = "default") -> Dict[str, Any]:
        """Extract all structured data from the current page."""
        data = await self.browser.extract_structured_data(context_id)
        page_state = await self.browser.get_page_state(context_id)

        return {
            'url': page_state.url,
            'title': page_state.title,
            'tables': data.get('tables', []),
            'lists': data.get('lists', []),
            'links': data.get('links', []),
            'headings': data.get('headings', []),
            'text_preview': page_state.content[:2000],
        }

    def to_csv(self, data: Dict) -> str:
        """Convert extracted data to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Export tables
        tables = data.get('tables', [])
        if tables:
            for i, table in enumerate(tables):
                writer.writerow([f"--- Table {i+1} ---"])
                for row in table.get('rows', []):
                    writer.writerow(row)
                writer.writerow([])

        # Export links
        links = data.get('links', [])
        if links:
            writer.writerow(["--- Links ---"])
            writer.writerow(["Text", "URL"])
            for link in links:
                writer.writerow([link.get('text', ''), link.get('href', '')])

        # Export lists
        lists = data.get('lists', [])
        if lists:
            writer.writerow([])
            writer.writerow(["--- Lists ---"])
            for i, lst in enumerate(lists):
                writer.writerow([f"List {i+1}"])
                for item in lst.get('items', []):
                    writer.writerow([item])

        return output.getvalue()

    def to_json(self, data: Dict) -> str:
        """Export data as formatted JSON."""
        return json.dumps(data, indent=2, ensure_ascii=False)

    def to_markdown(self, data: Dict) -> str:
        """Export data as Markdown text."""
        lines = [f"# {data.get('title', 'Extracted Data')}",
                 f"**URL:** {data.get('url', '')}",
                 ""]

        # Headings
        headings = data.get('headings', [])
        if headings:
            lines.append("## Page Structure")
            for h in headings:
                level = int(h.get('level', 'H2').replace('H', ''))
                lines.append(f"{'#' * level} {h.get('text', '')}")
            lines.append("")

        # Tables
        tables = data.get('tables', [])
        for i, table in enumerate(tables):
            lines.append(f"## Table {i+1}")
            rows = table.get('rows', [])
            if rows:
                # Header
                lines.append("| " + " | ".join(rows[0]) + " |")
                lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
                for row in rows[1:]:
                    padded = row + [''] * (len(rows[0]) - len(row))
                    lines.append("| " + " | ".join(padded) + " |")
            lines.append("")

        # Links
        links = data.get('links', [])
        if links:
            lines.append("## Links")
            for link in links[:30]:
                lines.append(f"- [{link.get('text', '')}]({link.get('href', '')})")
            lines.append("")

        return "\n".join(lines)
