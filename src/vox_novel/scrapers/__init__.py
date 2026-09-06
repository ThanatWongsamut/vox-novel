from vox_novel.scrapers.base import BaseScraper
from vox_novel.scrapers.registry import ScraperRegistry, registry
from vox_novel.scrapers.webnovel import WebnovelScraper

__all__ = ["BaseScraper", "ScraperRegistry", "registry", "WebnovelScraper"]
