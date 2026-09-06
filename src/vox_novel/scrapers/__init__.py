from vox_novel.scrapers.base import BaseScraper
from vox_novel.scrapers.registry import ScraperRegistry, registry
from vox_novel.scrapers.webnovel import WebnovelScraper
from vox_novel.scrapers.readtoon import ReadtoonScraper

__all__ = ["BaseScraper", "ScraperRegistry", "registry", "WebnovelScraper", "ReadtoonScraper"]

