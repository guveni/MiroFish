"""
OASIS simulation manager.
Manages parallel Twitter and Reddit simulations using preset scripts and
LLM-generated configuration parameters.
"""

import os
import json
import shutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.logger import get_logger
from ..utils.pipeline_retry import run_pipeline_step
from .run_checkpoint_store import (
    checkpoint_simulation_stage,
    load_simulation_stage_checkpoint,
)
from .zep_entity_reader import ZepEntityReader, FilteredEntities
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from .simulation_config_generator import (
    AgentActivityConfig,
    EventConfig,
    PlatformConfig,
    SimulationConfigGenerator,
    SimulationParameters,
    TimeSimulationConfig,
)
from ..utils.locale import t

logger = get_logger('mirofish.simulation')


class SimulationStatus(str, Enum):
    """Simulation status."""
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class PlatformType(str, Enum):
    """Platform type."""
    TWITTER = "twitter"
    REDDIT = "reddit"


@dataclass
class SimulationState:
    """Simulation state."""
    simulation_id: str
    project_id: str
    graph_id: str
    
    # Enabled platforms.
    enable_twitter: bool = True
    enable_reddit: bool = True
    
    # Status.
    status: SimulationStatus = SimulationStatus.CREATED
    
    # Preparation data.
    entities_count: int = 0
    profiles_count: int = 0
    entity_types: List[str] = field(default_factory=list)
    
    # Configuration generation info.
    config_generated: bool = False
    config_reasoning: str = ""
    
    # Runtime data.
    current_round: int = 0
    twitter_status: str = "not_started"
    reddit_status: str = "not_started"
    
    # Timestamps.
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Error info.
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Full internal state dictionary."""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "enable_twitter": self.enable_twitter,
            "enable_reddit": self.enable_reddit,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "config_reasoning": self.config_reasoning,
            "current_round": self.current_round,
            "twitter_status": self.twitter_status,
            "reddit_status": self.reddit_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }
    
    def to_simple_dict(self) -> Dict[str, Any]:
        """Simplified state dictionary for API responses."""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "error": self.error,
        }


class SimulationManager:
    """
    Simulation manager.
    
    Responsibilities:
    1. Read and filter entities from the Zep graph.
    2. Generate OASIS Agent Profiles.
    3. Generate simulation parameters with the LLM.
    4. Prepare the files required by preset scripts.
    """
    
    # Simulation data directory.
    SIMULATION_DATA_DIR = os.path.join(
        os.path.dirname(__file__), 
        '../../uploads/simulations'
    )
    
    def __init__(self):
        # Ensure the directory exists.
        os.makedirs(self.SIMULATION_DATA_DIR, exist_ok=True)
        
        # In-memory simulation state cache.
        self._simulations: Dict[str, SimulationState] = {}
    
    def _get_simulation_dir(self, simulation_id: str) -> str:
        """Return the simulation data directory."""
        sim_dir = os.path.join(self.SIMULATION_DATA_DIR, simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        return sim_dir
    
    def _save_simulation_state(self, state: SimulationState):
        """Save simulation state to disk."""
        sim_dir = self._get_simulation_dir(state.simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        state.updated_at = datetime.now().isoformat()
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        
        self._simulations[state.simulation_id] = state
    
    def _load_simulation_state(self, simulation_id: str) -> Optional[SimulationState]:
        """Load simulation state from disk."""
        if simulation_id in self._simulations:
            return self._simulations[simulation_id]
        
        sim_dir = self._get_simulation_dir(simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        if not os.path.exists(state_file):
            return None
        
        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        state = SimulationState(
            simulation_id=simulation_id,
            project_id=data.get("project_id", ""),
            graph_id=data.get("graph_id", ""),
            enable_twitter=data.get("enable_twitter", True),
            enable_reddit=data.get("enable_reddit", True),
            status=SimulationStatus(data.get("status", "created")),
            entities_count=data.get("entities_count", 0),
            profiles_count=data.get("profiles_count", 0),
            entity_types=data.get("entity_types", []),
            config_generated=data.get("config_generated", False),
            config_reasoning=data.get("config_reasoning", ""),
            current_round=data.get("current_round", 0),
            twitter_status=data.get("twitter_status", "not_started"),
            reddit_status=data.get("reddit_status", "not_started"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            error=data.get("error"),
        )
        
        self._simulations[simulation_id] = state
        return state

    @staticmethod
    def _entity_node_from_dict(data: Dict[str, Any]):
        from .zep_entity_reader import EntityNode

        return EntityNode(
            uuid=data.get("uuid", ""),
            name=data.get("name", ""),
            labels=data.get("labels", []),
            summary=data.get("summary", ""),
            attributes=data.get("attributes", {}),
            related_edges=data.get("related_edges", []),
            related_nodes=data.get("related_nodes", []),
        )

    @classmethod
    def _filtered_entities_from_payload(
        cls,
        payload: Dict[str, Any],
    ) -> Optional[FilteredEntities]:
        data = payload.get("filtered_entities")
        if not isinstance(data, dict):
            return None
        entities = [
            cls._entity_node_from_dict(item)
            for item in data.get("entities", [])
            if isinstance(item, dict)
        ]
        return FilteredEntities(
            entities=entities,
            entity_types=set(data.get("entity_types", [])),
            total_count=int(data.get("total_count", len(entities))),
            filtered_count=int(data.get("filtered_count", len(entities))),
        )

    @staticmethod
    def _profile_from_dict(data: Dict[str, Any]) -> OasisAgentProfile:
        allowed = OasisAgentProfile.__dataclass_fields__.keys()
        return OasisAgentProfile(**{k: v for k, v in data.items() if k in allowed})

    @staticmethod
    def _simulation_params_from_payload(
        payload: Dict[str, Any],
    ) -> Optional[SimulationParameters]:
        data = payload.get("simulation_parameters")
        if not isinstance(data, dict):
            return None
        return SimulationParameters(
            simulation_id=data.get("simulation_id", ""),
            project_id=data.get("project_id", ""),
            graph_id=data.get("graph_id", ""),
            simulation_requirement=data.get("simulation_requirement", ""),
            time_config=TimeSimulationConfig(**data.get("time_config", {})),
            agent_configs=[
                AgentActivityConfig(**item)
                for item in data.get("agent_configs", [])
                if isinstance(item, dict)
            ],
            event_config=EventConfig(**data.get("event_config", {})),
            twitter_config=(
                PlatformConfig(**data["twitter_config"])
                if isinstance(data.get("twitter_config"), dict)
                else None
            ),
            reddit_config=(
                PlatformConfig(**data["reddit_config"])
                if isinstance(data.get("reddit_config"), dict)
                else None
            ),
            llm_model=data.get("llm_model", ""),
            llm_base_url=data.get("llm_base_url", ""),
            generated_at=data.get("generated_at", datetime.now().isoformat()),
            generation_reasoning=data.get("generation_reasoning", ""),
        )
    
    def create_simulation(
        self,
        project_id: str,
        graph_id: str,
        enable_twitter: bool = True,
        enable_reddit: bool = True,
    ) -> SimulationState:
        """
        Create a new simulation.
        
        Args:
            project_id: Project ID.
            graph_id: Zep graph ID.
            enable_twitter: Whether Twitter simulation is enabled.
            enable_reddit: Whether Reddit simulation is enabled.
            
        Returns:
            SimulationState
        """
        import uuid
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"
        
        state = SimulationState(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit,
            status=SimulationStatus.CREATED,
        )
        
        self._save_simulation_state(state)
        logger.info("Created simulation: %s, project=%s, graph=%s", simulation_id, project_id, graph_id)
        
        return state
    
    def prepare_simulation(
        self,
        simulation_id: str,
        simulation_requirement: str,
        document_text: str,
        defined_entity_types: Optional[List[str]] = None,
        use_llm_for_profiles: bool = True,
        progress_callback: Optional[callable] = None,
        parallel_profile_count: int = 3
    ) -> SimulationState:
        """
        Prepare the simulation environment end to end.
        
        Steps:
        1. Read and filter entities from the Zep graph.
        2. Generate OASIS Agent Profiles, optionally LLM-enhanced and parallelized.
        3. Generate simulation configuration.
        4. Save configuration and profile files.
        5. Prepare the preset script environment.
        
        Args:
            simulation_id: Simulation ID.
            simulation_requirement: Requirement used for LLM-generated config.
            document_text: Source text for LLM context.
            defined_entity_types: Optional predefined entity types.
            use_llm_for_profiles: Whether to use LLM-generated detailed personas.
            progress_callback: Progress callback(stage, progress, message).
            parallel_profile_count: Number of concurrent persona workers.
            
        Returns:
            SimulationState
        """
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation does not exist: {simulation_id}")
        
        try:
            state.status = SimulationStatus.PREPARING
            self._save_simulation_state(state)
            
            sim_dir = self._get_simulation_dir(simulation_id)
            resume_enabled = Config.RESUME_FROM_CHECKPOINT
            
            # Stage 1: read and filter entities.
            if progress_callback:
                progress_callback("reading", 0, t('progress.connectingZepGraph'))
            
            filtered = None
            if resume_enabled:
                checkpoint = load_simulation_stage_checkpoint(
                    simulation_id,
                    "simulation_entities_loaded",
                )
                if checkpoint:
                    filtered = self._filtered_entities_from_payload(checkpoint["payload"])
                    if filtered:
                        logger.info(
                            "Resumed simulation entities from checkpoint: %s",
                            simulation_id,
                        )

            if filtered is None:
                reader = ZepEntityReader()
                
                if progress_callback:
                    progress_callback("reading", 30, t('progress.readingNodeData'))
                
                filtered = run_pipeline_step(
                    "zep_filter_entities_for_simulation",
                    lambda: reader.filter_defined_entities(
                        graph_id=state.graph_id,
                        defined_entity_types=defined_entity_types,
                        enrich_with_edges=True,
                    ),
                )
            
            state.entities_count = filtered.filtered_count
            state.entity_types = list(filtered.entity_types)
            
            if progress_callback:
                progress_callback(
                    "reading", 100,
                    t('progress.readingComplete', count=filtered.filtered_count),
                    current=filtered.filtered_count,
                    total=filtered.filtered_count
                )
            
            if filtered.filtered_count == 0:
                state.status = SimulationStatus.FAILED
                state.error = "No qualifying entities found; verify the graph was built correctly."
                self._save_simulation_state(state)
                return state

            try:
                checkpoint_simulation_stage(
                    simulation_id,
                    state.project_id,
                    "simulation_entities_loaded",
                    {
                        "simulation_id": simulation_id,
                        "graph_id": state.graph_id,
                        "entities_count": state.entities_count,
                        "entity_types": state.entity_types,
                        "filtered_entities": filtered.to_dict(),
                    },
                )
            except Exception as cp_err:
                logger.warning("checkpoint simulation_entities_loaded skipped: %s", cp_err)
            
            # Stage 2: generate Agent Profiles.
            total_entities = len(filtered.entities)
            
            if progress_callback:
                progress_callback(
                    "generating_profiles", 0,
                    t('progress.startGenerating'),
                    current=0,
                    total=total_entities
                )
            
            # Pass graph_id to enable richer Zep search context.
            generator = OasisProfileGenerator(graph_id=state.graph_id)
            
            def profile_progress(current, total, msg):
                if progress_callback:
                    progress_callback(
                        "generating_profiles", 
                        int(current / total * 100), 
                        msg,
                        current=current,
                        total=total,
                        item_name=msg
                    )
            
            # Set the incremental output path, preferring Reddit JSON.
            realtime_output_path = None
            realtime_platform = "reddit"
            if state.enable_reddit:
                realtime_output_path = os.path.join(sim_dir, "reddit_profiles.json")
                realtime_platform = "reddit"
            elif state.enable_twitter:
                realtime_output_path = os.path.join(sim_dir, "twitter_profiles.csv")
                realtime_platform = "twitter"
            
            profiles = None
            if resume_enabled:
                checkpoint = load_simulation_stage_checkpoint(
                    simulation_id,
                    "simulation_profiles_written",
                )
                if checkpoint:
                    raw_profiles = checkpoint["payload"].get("profiles")
                    if isinstance(raw_profiles, list):
                        profiles = [
                            self._profile_from_dict(item)
                            for item in raw_profiles
                            if isinstance(item, dict)
                        ]
                        logger.info(
                            "Resumed %d profiles from checkpoint: %s",
                            len(profiles),
                            simulation_id,
                        )

            if profiles is None:
                profiles = generator.generate_profiles_from_entities(
                    entities=filtered.entities,
                    use_llm=use_llm_for_profiles,
                    progress_callback=profile_progress,
                    graph_id=state.graph_id,
                    parallel_count=parallel_profile_count,
                    realtime_output_path=realtime_output_path,
                    output_platform=realtime_platform,
                )
            
            state.profiles_count = len(profiles)
            
            # Save profile files. Twitter uses CSV; Reddit uses JSON.
            # Reddit was also written incrementally, but save again for completeness.
            if progress_callback:
                progress_callback(
                    "generating_profiles", 95,
                    t('progress.savingProfiles'),
                    current=total_entities,
                    total=total_entities
                )
            
            if state.enable_reddit:
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "reddit_profiles.json"),
                    platform="reddit"
                )
            
            if state.enable_twitter:
                # Twitter uses CSV as required by OASIS.
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "twitter_profiles.csv"),
                    platform="twitter"
                )
            
            if progress_callback:
                progress_callback(
                    "generating_profiles", 100,
                    t('progress.profilesComplete', count=len(profiles)),
                    current=len(profiles),
                    total=len(profiles)
                )

            try:
                checkpoint_simulation_stage(
                    simulation_id,
                    state.project_id,
                    "simulation_profiles_written",
                    {
                        "simulation_id": simulation_id,
                        "graph_id": state.graph_id,
                        "profiles_count": state.profiles_count,
                        "sim_dir": sim_dir,
                        "profiles": [p.to_dict() for p in profiles],
                    },
                )
            except Exception as cp_err:
                logger.warning("checkpoint simulation_profiles_written skipped: %s", cp_err)
            
            # Stage 3: generate simulation configuration with the LLM.
            if progress_callback:
                progress_callback(
                    "generating_config", 0,
                    t('progress.analyzingRequirements'),
                    current=0,
                    total=3
                )
            
            config_generator = SimulationConfigGenerator()
            
            if progress_callback:
                progress_callback(
                    "generating_config", 30,
                    t('progress.callingLLMConfig'),
                    current=1,
                    total=3
                )
            
            sim_params = None
            if resume_enabled:
                checkpoint = load_simulation_stage_checkpoint(
                    simulation_id,
                    "simulation_config_generated",
                )
                if checkpoint:
                    sim_params = self._simulation_params_from_payload(checkpoint["payload"])
                    if sim_params:
                        logger.info(
                            "Resumed simulation config from checkpoint: %s",
                            simulation_id,
                        )

            if sim_params is None:
                sim_params = config_generator.generate_config(
                    simulation_id=simulation_id,
                    project_id=state.project_id,
                    graph_id=state.graph_id,
                    simulation_requirement=simulation_requirement,
                    document_text=document_text,
                    entities=filtered.entities,
                    enable_twitter=state.enable_twitter,
                    enable_reddit=state.enable_reddit,
                )

                try:
                    checkpoint_simulation_stage(
                        simulation_id,
                        state.project_id,
                        "simulation_config_generated",
                        {
                            "simulation_id": simulation_id,
                            "graph_id": state.graph_id,
                            "simulation_parameters": sim_params.to_dict(),
                        },
                    )
                except Exception as cp_err:
                    logger.warning("checkpoint simulation_config_generated skipped: %s", cp_err)
            
            if progress_callback:
                progress_callback(
                    "generating_config", 70,
                    t('progress.savingConfigFiles'),
                    current=2,
                    total=3
                )
            
            # Save configuration file.
            config_path = os.path.join(sim_dir, "simulation_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(sim_params.to_json())
            
            state.config_generated = True
            state.config_reasoning = sim_params.generation_reasoning
            
            if progress_callback:
                progress_callback(
                    "generating_config", 100,
                    t('progress.configComplete'),
                    current=3,
                    total=3
                )
            
            # Runner scripts remain in backend/scripts and are launched from there.
            
            # Update state.
            state.status = SimulationStatus.READY
            self._save_simulation_state(state)

            try:
                checkpoint_simulation_stage(
                    simulation_id,
                    state.project_id,
                    "simulation_prepared",
                    {
                        "simulation_id": simulation_id,
                        "graph_id": state.graph_id,
                        "entities_count": state.entities_count,
                        "profiles_count": state.profiles_count,
                        "config_path": config_path,
                        "config_reasoning_preview": (
                            state.config_reasoning[:2000]
                            if state.config_reasoning
                            else ""
                        ),
                    },
                )
            except Exception as cp_err:
                logger.warning("checkpoint simulation_prepared skipped: %s", cp_err)
            
            logger.info(
                "Simulation prepared: %s, entities=%d, profiles=%d",
                simulation_id,
                state.entities_count,
                state.profiles_count,
            )
            
            return state
            
        except Exception as e:
            logger.error("Simulation preparation failed: %s, error=%s", simulation_id, str(e))
            import traceback
            logger.error(traceback.format_exc())
            state.status = SimulationStatus.FAILED
            state.error = str(e)
            self._save_simulation_state(state)
            if getattr(state, "project_id", ""):
                try:
                    checkpoint_simulation_stage(
                        simulation_id,
                        state.project_id,
                        "simulation_prepare_failed",
                        {"simulation_id": simulation_id},
                        error=str(e),
                    )
                except Exception as cp_err:
                    logger.warning("checkpoint simulation_prepare_failed skipped: %s", cp_err)
            raise
    
    def get_simulation(self, simulation_id: str) -> Optional[SimulationState]:
        """Return simulation state."""
        return self._load_simulation_state(simulation_id)
    
    def list_simulations(self, project_id: Optional[str] = None) -> List[SimulationState]:
        """List simulations."""
        simulations = []
        
        if os.path.exists(self.SIMULATION_DATA_DIR):
            for sim_id in os.listdir(self.SIMULATION_DATA_DIR):
                # Skip hidden files such as .DS_Store and non-directories.
                sim_path = os.path.join(self.SIMULATION_DATA_DIR, sim_id)
                if sim_id.startswith('.') or not os.path.isdir(sim_path):
                    continue
                
                state = self._load_simulation_state(sim_id)
                if state:
                    if project_id is None or state.project_id == project_id:
                        simulations.append(state)
        
        return simulations
    
    def get_profiles(self, simulation_id: str, platform: str = "reddit") -> List[Dict[str, Any]]:
        """Return Agent Profiles for a simulation."""
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation does not exist: {simulation_id}")
        
        sim_dir = self._get_simulation_dir(simulation_id)
        profile_path = os.path.join(sim_dir, f"{platform}_profiles.json")
        
        if not os.path.exists(profile_path):
            return []
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_simulation_config(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Return simulation configuration."""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_run_instructions(self, simulation_id: str) -> Dict[str, str]:
        """Return run instructions."""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
        
        return {
            "simulation_dir": sim_dir,
            "scripts_dir": scripts_dir,
            "config_file": config_path,
            "commands": {
                "twitter": f"python {scripts_dir}/run_twitter_simulation.py --config {config_path}",
                "reddit": f"python {scripts_dir}/run_reddit_simulation.py --config {config_path}",
                "parallel": f"python {scripts_dir}/run_parallel_simulation.py --config {config_path}",
            },
            "instructions": (
                f"1. Activate your env: conda activate MiroFish (or equivalent)\n"
                f"2. Run simulators from {scripts_dir}:\n"
                f"   - Twitter only: python {scripts_dir}/run_twitter_simulation.py --config {config_path}\n"
                f"   - Reddit only: python {scripts_dir}/run_reddit_simulation.py --config {config_path}\n"
                f"   - Both in parallel: python {scripts_dir}/run_parallel_simulation.py --config {config_path}"
            )
        }
