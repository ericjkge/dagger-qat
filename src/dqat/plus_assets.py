"""Materialize byte-identical official assets needed by a fixed pilot."""

import hashlib
import shutil
import zlib
from pathlib import Path, PurePosixPath


class AssetDepot:
    def __init__(self, archive, root):
        self.archive, self.root = archive, Path(root).resolve()
        members = archive.infolist()
        prefixes = {
            m.filename.split("/assets/", 1)[0] + "/assets/" for m in members if "/assets/" in m.filename
        }
        if len(prefixes) != 1:
            raise ValueError("Expected one official assets subtree")
        prefix = prefixes.pop()
        self.members, self.extracted = {}, {}
        for member in members:
            if not member.filename.startswith(prefix):
                raise ValueError("Archive entry outside assets subtree")
            relative = member.filename[len(prefix) :]
            parts = PurePosixPath(relative)
            if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
                raise ValueError("Unsafe archive path")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unexpected archive symlink")
            if not member.is_dir():
                if relative in self.members:
                    raise ValueError("Duplicate asset entry")
                self.members[relative] = member

    def ensure(self, filename):
        path = Path(filename).resolve()
        if not path.is_relative_to(self.root):
            return
        relative = path.relative_to(self.root).as_posix()
        if relative in self.extracted:
            return
        if relative not in self.members:
            raise FileNotFoundError(f"Required asset absent from pinned archive: {relative}")
        if path.is_file():
            existing = path.read_bytes()
            member = self.members[relative]
            if len(existing) == member.file_size and zlib.crc32(existing) == member.CRC:
                self.extracted[relative] = hashlib.sha256(existing).hexdigest()
                return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".materializing")
        with self.archive.open(self.members[relative]) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(path)
        self.extracted[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    def common_assets(self):
        # Large object/scene collections are loaded only when their XML refers to them.
        for relative in self.members:
            if relative.split("/", 1)[0] not in ("new_objects", "scenes"):
                self.ensure(self.root / relative)

    def prepare_cases(self, cases, bddl_root):
        import random
        import xml.etree.ElementTree as ET

        import numpy as np
        from libero.libero.envs.env_wrapper import ControlEnv
        from robosuite.models.base import MujocoXML
        from robosuite.utils.binding_utils import MjSim

        original_parse = ET.parse
        original_resolve = MujocoXML.resolve_asset_dependency
        original_from_xml = MjSim.from_xml_string
        original_from_xml_descriptor = MjSim.__dict__["from_xml_string"]

        def parse(source, *args, **kwargs):
            if isinstance(source, (str, Path)):
                self.ensure(source)
            return original_parse(source, *args, **kwargs)

        def resolve(model):
            original_resolve(model)
            for node in model.asset.findall("./*[@file]"):
                self.ensure(node.get("file"))

        def from_xml(cls, xml, *args, **kwargs):
            # Arenas can add/change texture paths after their initial XML load.
            # Materialize the final simulator's complete file references as well.
            for node in ET.fromstring(xml).iter():
                if node.get("file"):
                    self.ensure(node.get("file"))
            return original_from_xml(xml, *args, **kwargs)

        ET.parse = parse
        MujocoXML.resolve_asset_dependency = resolve
        MjSim.from_xml_string = classmethod(from_xml)
        try:
            for index, case in enumerate(cases):
                random.seed(case["seed"])
                np.random.seed(case["seed"])
                env = ControlEnv(
                    bddl_file_name=str(Path(bddl_root) / "libero_10" / (case["task"] + ".bddl")),
                    use_camera_obs=False,
                    has_offscreen_renderer=False,
                    has_renderer=False,
                )
                env.close()
                print(f"Asset preflight {index + 1}/{len(cases)}; {len(self.extracted)} files", flush=True)
        finally:
            ET.parse = original_parse
            MujocoXML.resolve_asset_dependency = original_resolve
            MjSim.from_xml_string = original_from_xml_descriptor
