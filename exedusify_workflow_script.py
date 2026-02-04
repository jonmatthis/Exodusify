# %% [markdown]
# ## 0. Setup
# Import the Python modules used throughout the notebook. Make sure you have already installed the packages listed in the README (pandas, numpy, mutagen, unidecode).

# %% [markdown]
# ### Package bootstrap
# Install any missing Python packages required by this workflow so the import cell succeeds even on a fresh environment.

# # %%
# import importlib
# import subprocess
# import sys

# REQUIRED_PACKAGES = {
#     "pandas": "pandas",
#     "numpy": "numpy",
#     "mutagen": "mutagen",
#     "unidecode": "Unidecode"
# }

# for module_name, install_name in REQUIRED_PACKAGES.items():
#     try:
#         importlib.import_module(module_name)
#     except ImportError:
#         print(f"Installing missing dependency: {install_name}")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
# print("Dependency check complete.")

# %%
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import math
import re
import shutil
import pandas as pd
import numpy as np
from mutagen import File as MutagenFile
from unidecode import unidecode

# %% [markdown]
# ## 1. Configuration
# Define key directories (relative to the repository root) and ensure the output folder for reports exists.

# %%
# Adjust these paths if you relocate folders.
REPO_ROOT = Path.cwd()
MUSIC_ROOT = REPO_ROOT / "Music"
SPOTIFY_PLAYLISTS = REPO_ROOT / "spotify_playlists"
SHOPPING_LIST_DIR = REPO_ROOT / "shopping_lists"
LIBRARY_INDEX_CSV = REPO_ROOT / "library_index.csv"
ADD_ROOT = REPO_ROOT / "Add"
PLAYLIST_EXPORT_DIR = REPO_ROOT / "Playlists"

SHOPPING_LIST_DIR.mkdir(exist_ok=True)
ADD_ROOT.mkdir(exist_ok=True)
MUSIC_ROOT.mkdir(exist_ok=True)
PLAYLIST_EXPORT_DIR.mkdir(exist_ok=True)
print(f"Repository root: {REPO_ROOT}")
print(f"Music library: {MUSIC_ROOT}")
print(f"Spotify playlist CSVs: {SPOTIFY_PLAYLISTS}")
print(f"Shopping/output directory: {SHOPPING_LIST_DIR}")
print(f"Add drop directory: {ADD_ROOT}")
print(f"Playlist export directory: {PLAYLIST_EXPORT_DIR}")

# %% [markdown]
# ## 2. Helper functions
# Canonicalization helpers keep matching consistent between Spotify exports and local audio metadata.

# %%
NON_ALNUM = re.compile(r"[^a-z0-9]+")
FEAT_PATTERN = re.compile(r"\(feat\..*?\)", re.IGNORECASE)
REMIX_PATTERN = re.compile(r"-\s*(remaster(ed)?|remix|edit|mix).*", re.IGNORECASE)
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wav', '.aiff'}
INVALID_PATH_CHARS = re.compile(r"[<>:\"/\\|?*]")


def canonicalize_string(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = unidecode(str(value))
    normalized = FEAT_PATTERN.sub("", normalized)
    normalized = REMIX_PATTERN.sub("", normalized)
    normalized = normalized.lower()
    normalized = NON_ALNUM.sub(" ", normalized)
    normalized = normalized.strip()
    return re.sub(r"\s+", " ", normalized)


def safe_path_component(value: Optional[str], fallback: str = "Unknown") -> str:
    candidate = str(value).strip() if value else fallback
    candidate = unidecode(candidate)
    candidate = INVALID_PATH_CHARS.sub("", candidate)
    candidate = candidate.replace('/', '-').replace('\\', '-')
    candidate = candidate.strip()
    return candidate or fallback


def primary_artist(artists_field: Optional[str]) -> str:
    if not artists_field or not isinstance(artists_field, str):
        return ""
    first = artists_field.split(';')[0]
    return first.strip()


def friendly_playlist_name(csv_path: Path) -> str:
    name = csv_path.stem.replace('_', ' ')
    return name.strip()


def duration_ms_from_audio(audio_obj) -> Optional[int]:
    if audio_obj and audio_obj.info and getattr(audio_obj.info, 'length', None):
        return int(round(audio_obj.info.length * 1000))
    return None

# %% [markdown]
# ## 4. Scan the music library
# Create or refresh an auditable `library_index.csv` capturing metadata for every audio file under `Music/`.

# %%
def scan_music_library(music_root: Path) -> pd.DataFrame:
    records = []
    if not music_root.exists():
        print(f"Music directory not found: {music_root}")
        return pd.DataFrame()

    for file_path in music_root.rglob('*'):
        if not file_path.is_file() or file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        try:
            audio = MutagenFile(file_path)
        except Exception as exc:
            print(f"Failed to read {file_path}: {exc}")
            audio = None

        tags = getattr(audio, 'tags', None) if audio else None
        artist_tag = None
        title_tag = None
        album_tag = None

        if tags:
            artist_tag = tags.get('TPE1') or tags.get('artist')
            title_tag = tags.get('TIT2') or tags.get('title')
            album_tag = tags.get('TALB') or tags.get('album')

        artist_str = str(artist_tag.text[0]) if hasattr(artist_tag, 'text') else (artist_tag if isinstance(artist_tag, str) else None)
        title_str = str(title_tag.text[0]) if hasattr(title_tag, 'text') else (title_tag if isinstance(title_tag, str) else None)
        album_str = str(album_tag.text[0]) if hasattr(album_tag, 'text') else (album_tag if isinstance(album_tag, str) else None)

        # Fallbacks from the path structure
        if not artist_str:
            artist_str = file_path.parent.name
        if not title_str:
            title_str = file_path.stem

        records.append({
            'file_path': file_path.relative_to(music_root).as_posix(),
            'artist_raw': artist_str,
            'title_raw': title_str,
            'album_raw': album_str,
            'artist_canonical': canonicalize_string(artist_str),
            'title_canonical': canonicalize_string(title_str),
            'duration_ms': duration_ms_from_audio(audio)
        })

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.sort_values(['artist_canonical', 'title_canonical', 'file_path'], inplace=True)
    return df

library_index = scan_music_library(MUSIC_ROOT)
print(f"Indexed {len(library_index):,} local tracks")
if not library_index.empty:
    library_index.to_csv(LIBRARY_INDEX_CSV, index=False)
    display(library_index.head())
else:
    print('Library index is empty – check MUSIC_ROOT or file extensions.')

spotify_df = library_index

# %% [markdown]
# ## 5. Load Spotify playlist exports
# Combine all CSV files in `spotify_playlists/` into a single DataFrame with helpful flags.

# %%
DURATION_TOLERANCE_MS = 3000

def match_tracks(spotify_df: pd.DataFrame, library_df: pd.DataFrame, duration_tolerance_ms: int = DURATION_TOLERANCE_MS) -> pd.DataFrame:
    if spotify_df.empty:
        return pd.DataFrame()
    if library_df.empty:
        result = spotify_df.copy()
        result['file_path'] = pd.NA
        result['duration_ms_local'] = pd.NA
        return result

    lib_cols = library_df.rename(columns={'duration_ms': 'duration_ms_local'})
    merged = spotify_df.merge(lib_cols, how='left', on=['artist_canonical', 'title_canonical'], suffixes=('_spotify', '_local'))

    if 'duration_ms_local' in merged.columns:
        mask = merged['duration_ms_local'].notna() & merged['Duration (ms)'].notna()
        mismatched = mask & (merged['Duration (ms)'] - merged['duration_ms_local']).abs() > duration_tolerance_ms
        merged.loc[mismatched, ['file_path', 'duration_ms_local']] = pd.NA
    return merged

matched_df = match_tracks(spotify_df, library_index)
print(f"Matched rows: {len(matched_df):,}")
if not matched_df.empty:
    have_files = matched_df['file_path'].notna().sum()
    print(f"Tracks already downloaded: {have_files:,}")
    print(f"Tracks missing locally: {len(matched_df) - have_files:,}")
    display(matched_df.head())

# %% [markdown]
# ## 7. Build a missing-tracks shopping list
# Highlight Spotify tracks that are still missing from `Music/` so you can go hunt them down.

# %%
def build_shopping_list(matched_df: pd.DataFrame) -> pd.DataFrame:
    if matched_df.empty:
        return pd.DataFrame()
    missing = matched_df[matched_df['file_path'].isna()].copy()
    if missing.empty:
        return pd.DataFrame()

    grouped = (
        missing.groupby(['artist_canonical', 'title_canonical'], as_index=False)
        .agg({
            'primary_artist': 'first',
            'Track Name': 'first',
            'Album Name': lambda col: col.dropna().iloc[0] if col.dropna().any() else pd.NA,
            'Duration (ms)': 'first',
            'playlist_name': lambda col: sorted(set(col)),
            'is_liked': 'any',
            'is_top_songs': 'any'
        })
    )
    grouped['Playlists_Count'] = grouped['playlist_name'].apply(len)
    grouped['Playlists'] = grouped['playlist_name'].apply(lambda names: '; '.join(names))
    grouped.rename(columns={
        'primary_artist': 'Artist',
        'Track Name': 'Title',
        'Album Name': 'Album',
        'Duration (ms)': 'Duration_ms',
        'is_liked': 'Is_Liked',
        'is_top_songs': 'Is_Top_Songs'
    }, inplace=True)
    columns = ['Artist', 'Title', 'Album', 'Duration_ms', 'Playlists_Count', 'Playlists', 'Is_Liked', 'Is_Top_Songs']
    grouped = grouped[columns]
    grouped.sort_values(['Playlists_Count', 'Is_Liked', 'Artist', 'Title'], ascending=[False, False, True, True], inplace=True)
    return grouped

shopping_df = build_shopping_list(matched_df)
if shopping_df.empty:
    print('All playlist tracks already exist locally – no shopping list generated.')
else:
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    shopping_path = SHOPPING_LIST_DIR / f'shopping_list_{timestamp}.csv'
    shopping_df.to_csv(shopping_path, index=False)
    print(f"Shopping list saved to {shopping_path}")
    display(shopping_df.head())

# %% [markdown]
# ## 8. Generate an orphaned-tracks list
# Highlight tracks that exist in `Music/` but are not referenced by any current playlist snapshot.

# %%
def build_orphaned_tracks(matched_df: pd.DataFrame, library_df: pd.DataFrame) -> pd.DataFrame:
    if library_df.empty:
        return pd.DataFrame()
    playlist_keys = set(zip(matched_df['artist_canonical'], matched_df['title_canonical'])) if not matched_df.empty else set()
    library_df = library_df.copy()
    library_df['key'] = list(zip(library_df['artist_canonical'], library_df['title_canonical']))
    mask = library_df['key'].apply(lambda key: key not in playlist_keys)
    orphaned = library_df[mask].copy()
    if orphaned.empty:
        return pd.DataFrame()
    orphaned.rename(columns={
        'artist_raw': 'Artist',
        'title_raw': 'Title',
        'album_raw': 'Album',
        'duration_ms': 'Duration_ms'
    }, inplace=True)
    columns = ['Artist', 'Title', 'Album', 'Duration_ms', 'file_path']
    return orphaned[columns]

orphan_df = build_orphaned_tracks(matched_df, library_index)
if orphan_df.empty:
    print('No orphaned tracks – every local track appears in at least one playlist snapshot.')
else:
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    orphan_path = SHOPPING_LIST_DIR / f'orphaned_tracks_{timestamp}.csv'
    orphan_df.to_csv(orphan_path, index=False)
    print(f"Orphaned-track report saved to {orphan_path}")
    display(orphan_df.head())

# %% [markdown]
# ## 9. Show Playlist Statistics
# Summarize key statistics about each playlist, including total tracks, matched tracks, missing tracks, and orphaned tracks.

# %%
if 'matched_df' not in globals():
    print("Run cells 1-7 to create 'matched_df' before generating playlist stats.")
elif matched_df.empty:
    print('Playlist DataFrame is empty – load Spotify CSVs first.')
else:
    stats = (
        matched_df
        .groupby('playlist_name', dropna=False)
        .agg(
            Total_Tracks=('Track Name', 'size'),
            Matched_Tracks=('file_path', lambda col: col.notna().sum()),
            Liked_Snapshot=('is_liked', 'any'),
            Top_Songs_Snapshot=('is_top_songs', 'any')
        )
        .reset_index()
    )
    stats['Missing_Tracks'] = stats['Total_Tracks'] - stats['Matched_Tracks']
    stats['Percent_Complete'] = (stats['Matched_Tracks'] / stats['Total_Tracks'] * 100).round(1)
    stats.sort_values(['Percent_Complete', 'playlist_name'], ascending=[False, True], inplace=True)

    overall_total = int(stats['Total_Tracks'].sum())
    overall_missing = int(stats['Missing_Tracks'].sum())
    overall_matched = overall_total - overall_missing

    missing_unique = (
        matched_df[matched_df['file_path'].isna()]
        .drop_duplicates(subset=['artist_canonical', 'title_canonical'])
        .shape[0]
    )

    print(
        f"Playlists analyzed: {len(stats)} | Tracks: {overall_total:,} | "
        f"Matched: {overall_matched:,} | Missing: {overall_missing:,}"
    )
    print(f"Unique missing tracks across all playlists: {missing_unique:,}")
    display(stats)


# %% [markdown]
# ## 10. Export device playlists
# Write `.m3u8` files under `Playlists/` so the Innioasis Y1 can load each Spotify playlist. Paths are written as `../Music/...` so the playlist remains valid once copied next to the `Music/` folder.

# %%
def export_playlists(matched_df: pd.DataFrame, export_dir: Path, music_root: Path) -> pd.DataFrame:
    """Write .m3u8 playlists for every Spotify snapshot using ../Music/ paths."""
    if matched_df is None or matched_df.empty:
        print('Matched Spotify data is empty – load playlists and run the matcher before exporting.')
        return pd.DataFrame()

    export_dir.mkdir(parents=True, exist_ok=True)
    exports: list[dict[str, object]] = []
    missing_tracks = 0
    music_folder_name = music_root.name

    for playlist_name, group in matched_df.groupby('playlist_name', dropna=False):
        display_name = playlist_name if pd.notna(playlist_name) else 'Unnamed Playlist'
        human_name = display_name.replace('_', ' ')
        safe_name = safe_path_component(human_name, 'Playlist')
        playlist_path = export_dir / f"{safe_name}.m3u8"
        written = 0

        with playlist_path.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write('#EXTM3U\n')
            for _, row in group.iterrows():
                track_rel = row.get('file_path')
                if pd.isna(track_rel):
                    missing_tracks += 1
                    continue

                artist = row.get('primary_artist') or row.get('Artist Name(s)') or 'Unknown Artist'
                title = row.get('Track Name') or row.get('title') or 'Unknown Title'
                duration_ms = row.get('Duration (ms)')
                if pd.isna(duration_ms):
                    duration_ms = row.get('duration_ms_local')
                duration_seconds = (
                    int(round(float(duration_ms) / 1000)) if pd.notna(duration_ms) else -1
                )

                handle.write(f"#EXTINF:{duration_seconds},{artist} - {title}\n")

                normalized_rel = str(track_rel).replace('\\', '/').lstrip('/')
                music_prefix = f"{music_folder_name.lower()}/"
                if normalized_rel.lower().startswith(music_prefix):
                    normalized_rel = normalized_rel[len(music_prefix):]
                handle.write(f"../{music_folder_name}/{normalized_rel}\n")
                written += 1

        exports.append(
            {
                'playlist_name': human_name,
                'playlist_file': playlist_path.relative_to(export_dir.parent),
                'tracks_written': written,
            }
        )

    export_log = pd.DataFrame(exports)
    if not export_log.empty:
        display(export_log.sort_values('playlist_name'))
    print(f"Exported {len(exports)} playlists to {export_dir}")
    if missing_tracks:
        print(f"Skipped {missing_tracks:,} tracks because the audio files are still missing.")
    return export_log


playlist_export_log = export_playlists(matched_df, PLAYLIST_EXPORT_DIR, MUSIC_ROOT)

# %% [markdown]
# ## 3. Process new additions
# Move fresh MP3 downloads from the staging `Add/` folder into the canonical `Music/Artist/Album/Title.mp3` structure before scanning the library.

# %%
def process_new_additions(add_root: Path, music_root: Path, playlist_csv_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move staged downloads from Add/ into Music/ and summarize the results."""
    SUPPORTED_IMPORT_SUFFIXES = {'.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wav'}
    PLAYLIST_STAGE_FOLDER = "To Playlist"

    add_root.mkdir(exist_ok=True)
    playlist_staging_root = add_root / PLAYLIST_STAGE_FOLDER
    playlist_staging_root.mkdir(parents=True, exist_ok=True)

    # Ensure playlist staging folders mirror the Spotify CSV exports for easy triage.
    staging_lookup: dict[Path, str] = {}
    playlist_csvs = sorted(playlist_csv_root.glob('*.csv')) if playlist_csv_root.exists() else []
    for csv_file in playlist_csvs:
        playlist_name = friendly_playlist_name(csv_file)
        staging_dir = playlist_staging_root / safe_path_component(playlist_name, 'Playlist')
        staging_lookup[staging_dir] = playlist_name
        staging_dir.mkdir(parents=True, exist_ok=True)
    if playlist_csvs:
        print(
            "Ensured playlist staging folders exist for "
            f"{len(playlist_csvs)} playlists under {playlist_staging_root}."
        )
    else:
        print(
            f"No Spotify CSVs detected in {playlist_csv_root} – add files under Add/ manually or export playlists first."
        )

    actions: list[dict[str, object]] = []
    add_files = sorted(add_root.rglob('*'))
    for file_path in add_files:
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES:
            continue
        action: dict[str, object] = {
            'source': file_path.relative_to(add_root).as_posix(),
            'destination': pd.NA,
            'playlist_target': pd.NA,
            'status': None,
            'reason': pd.NA,
            'artist': pd.NA,
            'album': pd.NA,
            'title': pd.NA,
        }

        if not file_path.exists():
            action['status'] = 'skipped_missing_source'
            action['reason'] = 'File disappeared before processing.'
            actions.append(action)
            continue

        try:
            rel_to_staging = file_path.relative_to(playlist_staging_root)
        except ValueError:
            rel_to_staging = None
        if rel_to_staging:
            rel_stage_parts = rel_to_staging.parts
            if rel_stage_parts:
                staging_dir = playlist_staging_root / rel_stage_parts[0]
                if staging_dir in staging_lookup:
                    action['playlist_target'] = staging_lookup[staging_dir]

        try:
            audio = MutagenFile(file_path)
        except Exception as exc:
            action['status'] = 'error_read'
            action['reason'] = str(exc)
            actions.append(action)
            continue

        tags = getattr(audio, 'tags', None) if audio else None
        artist_tag = tags.get('TPE1') or tags.get('artist') if tags else None
        title_tag = tags.get('TIT2') or tags.get('title') if tags else None
        album_tag = tags.get('TALB') or tags.get('album') if tags else None

        def tag_value(raw_tag) -> Optional[str]:
            if hasattr(raw_tag, 'text') and raw_tag.text:
                return str(raw_tag.text[0]).strip()
            if isinstance(raw_tag, str):
                return raw_tag.strip()
            return None

        artist_value = tag_value(artist_tag)
        album_value = tag_value(album_tag)
        title_value = tag_value(title_tag)

        rel_parent_parts = list(file_path.relative_to(add_root).parts)
        if rel_parent_parts and rel_parent_parts[0] == PLAYLIST_STAGE_FOLDER:
            rel_parent_parts = rel_parent_parts[1:]
        if not artist_value and rel_parent_parts:
            artist_value = rel_parent_parts[0]
        if not album_value and len(rel_parent_parts) > 1:
            album_value = rel_parent_parts[1]
        if not title_value:
            title_value = file_path.stem

        if not artist_value:
            action['status'] = 'skipped_unknown_artist'
            action['reason'] = 'Missing artist tag after fallbacks.'
            actions.append(action)
            continue
        if not album_value:
            action['status'] = 'skipped_unknown_album'
            action['reason'] = 'Missing album tag after fallbacks.'
            actions.append(action)
            continue
        if not title_value:
            action['status'] = 'skipped_missing_tags'
            action['reason'] = 'Unable to infer track title.'
            actions.append(action)
            continue

        artist_component = safe_path_component(artist_value, 'Unknown Artist')
        album_component = safe_path_component(album_value, 'Unknown Album')
        title_component = safe_path_component(title_value, file_path.stem)
        dest_dir = music_root / artist_component / album_component
        dest_path = dest_dir / f"{title_component}{file_path.suffix.lower()}"

        canonical_title = canonicalize_string(title_value)
        duplicate_target = None
        try:
            if dest_dir.exists():
                for existing_file in dest_dir.iterdir():
                    if not existing_file.is_file():
                        continue
                    existing_canonical = canonicalize_string(existing_file.stem)
                    if existing_canonical == canonical_title:
                        duplicate_target = existing_file
                        break
        except FileNotFoundError:
            # Destination dir may have been deleted between the exists() check and iteration
            pass

        if dest_path.exists():
            action['status'] = 'skipped_exists'
            action['reason'] = f"Destination already exists: {dest_path}"
            actions.append(action)
            continue
        if duplicate_target is not None:
            action['status'] = 'skipped_duplicate_title'
            action['reason'] = f"Similar track already present: {duplicate_target}"
            actions.append(action)
            continue

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(dest_path))
        except Exception as exc:
            action['status'] = 'error_move'
            action['reason'] = str(exc)
            actions.append(action)
            continue

        action['status'] = 'moved'
        action['destination'] = dest_path.relative_to(music_root).as_posix()
        action['artist'] = artist_value
        action['album'] = album_value
        action['title'] = title_value
        actions.append(action)

    result_df = pd.DataFrame(actions)

    playlist_updates_df = pd.DataFrame()
    if not result_df.empty:
        moved = result_df[result_df['status'] == 'moved']
        if not moved.empty:
            playlist_updates_df = (
                moved.groupby('playlist_target', dropna=True)
                .size()
                .reset_index(name='tracks_added')
                .rename(columns={'playlist_target': 'playlist_name'})
            )
            playlist_updates_df.sort_values('tracks_added', ascending=False, inplace=True)

    return result_df, playlist_updates_df


new_additions_log, playlist_updates_log = process_new_additions(ADD_ROOT, MUSIC_ROOT, SPOTIFY_PLAYLISTS)
if not new_additions_log.empty:
    display(new_additions_log)
if playlist_updates_log is not None and not playlist_updates_log.empty:
    print('Updated playlists based on Add/ staging folders:')
    display(playlist_updates_log)


