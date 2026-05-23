"""
Simulation configuration generator.
Uses the LLM to derive detailed simulation parameters from requirements,
documents, and graph entities.

The generator uses smaller steps to reduce long-output failures:
1. Time configuration
2. Event configuration
3. Batched Agent configuration
4. Platform configuration
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.simulation_config')

# Reference daily rhythm configuration for UTC+8.
CHINA_TIMEZONE_CONFIG = {
    # Late night hours with minimal activity.
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Morning hours with gradually increasing activity.
    "morning_hours": [6, 7, 8],
    # Work hours.
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Evening peak hours.
    "peak_hours": [19, 20, 21, 22],
    # Late evening hours with declining activity.
    "night_hours": [23],
    # Activity multipliers.
    "activity_multipliers": {
        "dead": 0.05,
        "morning": 0.4,
        "work": 0.7,
        "peak": 1.5,
        "night": 0.5,
    }
}


@dataclass
class AgentActivityConfig:
    """Activity configuration for one Agent."""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    # Overall activity level (0.0-1.0).
    activity_level: float = 0.5
    
    # Expected posting frequency per hour.
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    
    # Active hours in 24-hour time.
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    
    # Reaction delay to hot events, in simulated minutes.
    response_delay_min: int = 5
    response_delay_max: int = 60
    
    # Sentiment tendency from negative to positive.
    sentiment_bias: float = 0.0
    
    # Stance toward the topic.
    stance: str = "neutral"  # supportive, opposing, neutral, observer
    
    # Visibility/influence weight.
    influence_weight: float = 1.0


@dataclass  
class TimeSimulationConfig:
    """Time-flow configuration."""
    # Total simulated duration in hours.
    total_simulation_hours: int = 72
    
    # Simulated minutes per round.
    minutes_per_round: int = 60
    
    # Number of Agents activated per hour.
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    
    # Peak activity hours.
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    
    # Low-activity hours.
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05
    
    # Morning hours.
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4
    
    # Work hours.
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Event configuration."""
    # Trigger posts at simulation start.
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Events triggered at specific times.
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Hot-topic keywords.
    hot_topics: List[str] = field(default_factory=list)
    
    # Narrative direction.
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Platform-specific configuration."""
    platform: str  # twitter or reddit
    
    # Recommendation weights.
    recency_weight: float = 0.4
    popularity_weight: float = 0.3
    relevance_weight: float = 0.3
    
    # Interaction threshold for viral spread.
    viral_threshold: int = 10
    
    # Strength of same-view clustering.
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Complete simulation parameters."""
    # Basic identifiers.
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str
    
    # Time configuration.
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    
    # Agent configuration list.
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    
    # Event configuration.
    event_config: EventConfig = field(default_factory=EventConfig)
    
    # Platform configuration.
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    
    # LLM configuration.
    llm_model: str = ""
    llm_base_url: str = ""
    
    # Generation metadata.
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Simulation configuration generator.
    
    Uses the LLM to analyze simulation requirements, document content, and graph
    entities, then generates the best-fit simulation parameters.
    
    Step strategy:
    1. Generate time and event configuration.
    2. Generate Agent configuration in batches.
    3. Generate platform configuration.
    """
    
    # Maximum context length.
    MAX_CONTEXT_LENGTH = 50000
    # Agent count per generated batch.
    AGENTS_PER_BATCH = 15
    
    # Per-step context truncation lengths.
    TIME_CONFIG_CONTEXT_LENGTH = 10000
    EVENT_CONFIG_CONTEXT_LENGTH = 8000
    ENTITY_SUMMARY_LENGTH = 300
    AGENT_SUMMARY_LENGTH = 300
    ENTITIES_PER_TYPE_DISPLAY = 20
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.llm_client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
        )
        self.model_name = self.llm_client.model
        self.base_url = (
            self.llm_client.base_url
            or Config.AZURE_OPENAI_ENDPOINT
            or Config.LLM_BASE_URL
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        Generate complete simulation configuration in staged steps.
        
        Args:
            simulation_id: Simulation ID.
            project_id: Project ID.
            graph_id: Graph ID.
            simulation_requirement: Simulation requirement.
            document_text: Source document text.
            entities: Filtered entity list.
            enable_twitter: Whether Twitter simulation is enabled.
            enable_reddit: Whether Reddit simulation is enabled.
            progress_callback: Progress callback(current_step, total_steps, message).
            
        Returns:
            Complete simulation parameters.
        """
        logger.info(
            "Generating simulation config: simulation_id=%s, entity_count=%d",
            simulation_id,
            len(entities),
        )
        
        # Calculate total steps.
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches
        current_step = 0
        
        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")
        
        # Build shared context.
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        # ========== Steps 1-2: generate time and event config in parallel ==========
        num_entities = len(entities)
        report_progress(1, t('progress.generatingTimeConfig'))
        report_progress(2, t('progress.generatingEventConfig'))
        with ThreadPoolExecutor(max_workers=2) as executor:
            time_future = executor.submit(self._generate_time_config, context, num_entities)
            event_future = executor.submit(
                self._generate_event_config,
                context,
                simulation_requirement,
                entities,
            )
            time_config_result = time_future.result()
            event_config_result = event_future.result()

        time_config = self._parse_time_config(time_config_result, num_entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"{t('progress.timeConfigLabel')}: {time_config_result.get('reasoning', t('common.success'))}")
        reasoning_parts.append(f"{t('progress.eventConfigLabel')}: {event_config_result.get('reasoning', t('common.success'))}")
        
        # ========== Steps 3-N: generate Agent config batches in parallel ==========
        all_agent_configs_by_batch: List[List[AgentActivityConfig]] = [[] for _ in range(num_batches)]
        if num_batches:
            max_workers = min(max(1, Config.SIM_CONFIG_MAX_WORKERS), num_batches)
            use_async_batches = True
            try:
                import asyncio

                asyncio.get_running_loop()
                use_async_batches = False
            except RuntimeError:
                pass

            if use_async_batches:
                all_agent_configs_by_batch = asyncio.run(
                    self._generate_agent_config_batches_async(
                        context,
                        entities,
                        simulation_requirement,
                        num_batches,
                        max_workers,
                        report_progress,
                    )
                )
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {}
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * self.AGENTS_PER_BATCH
                        end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
                        batch_entities = entities[start_idx:end_idx]
                        future = executor.submit(
                            self._generate_agent_configs_batch,
                            context,
                            batch_entities,
                            start_idx,
                            simulation_requirement,
                        )
                        future_to_batch[future] = (batch_idx, start_idx, end_idx)

                    for future in as_completed(future_to_batch):
                        batch_idx, start_idx, end_idx = future_to_batch[future]
                        report_progress(
                            3 + batch_idx,
                            t('progress.generatingAgentConfig', start=start_idx + 1, end=end_idx, total=len(entities))
                        )
                        all_agent_configs_by_batch[batch_idx] = future.result()

        all_agent_configs = [
            cfg
            for batch_configs in all_agent_configs_by_batch
            for cfg in batch_configs
        ]
        
        reasoning_parts.append(t('progress.agentConfigResult', count=len(all_agent_configs)))
        
        # Assign poster Agents for initial posts.
        logger.info("Assigning poster Agents for initial posts...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(t('progress.postAssignResult', count=assigned_count))
        
        # Final step: platform configuration.
        report_progress(total_steps, t('progress.generatingPlatformConfig'))
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        # Build final parameters.
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info("Simulation config generated: %d Agent configs", len(params.agent_configs))
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """Build LLM context and truncate to the maximum length."""
        
        # Entity summary.
        entity_summary = self._summarize_entities(entities)
        
        # Context body.
        context_parts = [
            f"## Simulation Requirement\n{simulation_requirement}",
            f"\n## Entities ({len(entities)})\n{entity_summary}",
        ]
        
        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500
        
        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(document truncated)"
            context_parts.append(f"\n## Source Document Text\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Generate an entity summary."""
        lines = []
        
        # Group by entity type.
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)})")
            # Use configured display count and summary length.
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... {len(type_entities) - display_count} more")
        
        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Call the LLM and repair JSON when possible."""
        import re
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                content = self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),
                    max_tokens=Config.LLM_JSON_MAX_TOKENS,
                )

                # Try to parse JSON.
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, str(e)[:80])
                    
                    # Try to repair JSON.
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    
                    last_error = e
                    
            except Exception as e:
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, str(e)[:80])
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))
        
        raise last_error or Exception("LLM call failed")

    async def _call_llm_with_retry_async(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Async LLM call with the same JSON repair behavior."""
        import asyncio

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                content = await self.llm_client.achat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),
                    max_tokens=Config.LLM_JSON_MAX_TOKENS,
                )

                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, str(e)[:80])
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    last_error = e

            except Exception as e:
                logger.warning("Async LLM call failed (attempt %d): %s", attempt + 1, str(e)[:80])
                last_error = e
                await asyncio.sleep(2 * (attempt + 1))

        raise last_error or Exception("LLM call failed")
    
    def _fix_truncated_json(self, content: str) -> str:
        """Repair truncated JSON."""
        content = content.strip()
        
        # Count unclosed brackets.
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Close a likely unfinished string.
        if content and content[-1] not in '",}]':
            content += '"'
        
        # Close brackets.
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Try to repair configuration JSON."""
        import re
        
        # Repair truncation first.
        content = self._fix_truncated_json(content)
        
        # Extract the JSON section.
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # Remove newlines inside JSON string values.
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            
            try:
                return json.loads(json_str)
            except:
                # Try removing control characters.
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
        
        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generate time configuration."""
        # Use the configured context truncation length.
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]
        
        # Cap active Agents at 90% of the available population.
        max_agents_allowed = max(1, int(num_entities * 0.9))
        
        prompt = f"""Generate a time-flow configuration for the simulation requirement below.

{context_truncated}

## Task
Return time configuration JSON.

### Guidance
- Infer the target audience's likely time zone and daily rhythm from the scenario.
- The UTC+8 pattern below is only a reference: 00-05 minimal activity, 06-08 rising activity, 09-18 moderate workday activity, 19-22 peak activity, and lower activity after 23.
- Adjust specific hours for the event and population. Students may peak later, media may be active all day, official institutions may post only during work hours, and breaking news may keep late-night activity high.

### JSON format, no Markdown

Example:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Why this time configuration fits the event"
}}

Fields:
- total_simulation_hours (int): total duration, 24-168 hours.
- minutes_per_round (int): 30-120 minutes; 60 is usually a good default.
- agents_per_hour_min (int): range 1-{max_agents_allowed}.
- agents_per_hour_max (int): range 1-{max_agents_allowed}.
- peak_hours (int array): peak hours for this audience.
- off_peak_hours (int array): low-activity hours.
- morning_hours (int array): morning hours.
- work_hours (int array): workday hours.
- reasoning (string): brief rationale."""

        system_prompt = "You are a social media simulation expert. Return pure JSON only. The time configuration must fit the target audience's likely daily rhythm."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning("Time configuration LLM generation failed: %s; using defaults", e)
            return self._get_default_time_config(num_entities)
    
    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Return a default UTC+8-style time configuration."""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Using default UTC+8 daily rhythm with one simulated hour per round."
        }
    
    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """Parse time configuration and keep agents_per_hour within bounds."""
        # Raw values.
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))
        
        # Ensure values do not exceed the total Agent count.
        if agents_per_hour_min > num_entities:
            logger.warning(
                "agents_per_hour_min (%s) exceeded Agent count (%s); adjusted",
                agents_per_hour_min,
                num_entities,
            )
            agents_per_hour_min = max(1, num_entities // 10)
        
        if agents_per_hour_max > num_entities:
            logger.warning(
                "agents_per_hour_max (%s) exceeded Agent count (%s); adjusted",
                agents_per_hour_max,
                num_entities,
            )
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)
        
        # Ensure min < max.
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning("agents_per_hour_min >= max; adjusted to %s", agents_per_hour_min)
        
        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self, 
        context: str, 
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Generate event configuration."""
        
        # Available entity types for the LLM.
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))
        
        # Representative entity names by type.
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)
        
        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}" 
            for t, examples in type_examples.items()
        ])
        
        # Use configured context truncation length.
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]
        
        prompt = f"""Generate event configuration for this simulation requirement.

Simulation requirement: {simulation_requirement}

{context_truncated}

## Available entity types and examples
{type_info}

## Task
Return event configuration JSON:
- Extract hot-topic keywords.
- Describe the narrative direction.
- Design initial post content. Every post must specify poster_type.

Important: poster_type must be selected from the available entity types above so the post can be assigned to a suitable Agent. For example, official statements should come from Official/University types, news from MediaOutlet, and student viewpoints from Student.

Return JSON only, no Markdown:
{{
    "hot_topics": ["keyword1", "keyword2", ...],
    "narrative_direction": "<narrative direction>",
    "initial_posts": [
        {{"content": "post content", "poster_type": "entity type from the available list"}},
        ...
    ],
    "reasoning": "<brief rationale>"
}}"""

        system_prompt = "You are a public-opinion analysis expert. Return pure JSON only. poster_type must exactly match an available entity type."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'poster_type' field value MUST be in English PascalCase exactly matching the available entity types. Only 'content', 'narrative_direction', 'hot_topics' and 'reasoning' fields should use the specified language."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning("Event configuration LLM generation failed: %s; using defaults", e)
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Using default empty event configuration."
            }
    
    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Parse event configuration."""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Assign suitable poster Agents for initial posts.
        
        Matches each post's poster_type to the best available agent_id.
        """
        if not event_config.initial_posts:
            return event_config
        
        # Build an Agent index by entity type.
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)
        
        # Type aliases for slight LLM formatting differences.
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }
        
        # Track used Agents per type to avoid always selecting the first one.
        used_indices: Dict[str, int] = {}
        
        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")
            
            # Try to find a matching Agent.
            matched_agent_id = None
            
            # Direct match.
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # Alias match.
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break
            
            # Fallback to the most influential Agent.
            if matched_agent_id is None:
                logger.warning(
                    "No matching Agent found for type %r; using the most influential Agent",
                    poster_type,
                )
                if agent_configs:
                    # Sort by influence and choose the highest.
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0
            
            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })
            
            logger.info(
                "Initial post assignment: poster_type=%r -> agent_id=%s",
                poster_type,
                matched_agent_id,
            )
        
        event_config.initial_posts = updated_posts
        return event_config
    
    def _build_agent_config_batch_prompt(
        self,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> tuple[str, str]:
        """Build the prompt pair for one Agent configuration batch."""
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })
        
        prompt = f"""Generate social media activity configuration for each entity.

Simulation requirement: {simulation_requirement}

## Entity list
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Generate activity configuration for every entity. Tune timing to the target audience's likely daily rhythm.
- Official institutions such as University/GovernmentAgency: low activity (0.1-0.3), work-hour activity (9-17), slower response (60-240 minutes), high influence (2.5-3.0).
- MediaOutlet: moderate activity (0.4-0.6), broad daily activity (8-23), fast response (5-30 minutes), high influence (2.0-2.5).
- Individuals such as Student/Person/Alumni: high activity (0.6-0.9), mostly evening activity (18-23), fast response (1-15 minutes), lower influence (0.8-1.2).
- Public figures/experts: moderate activity (0.4-0.6), medium-high influence (1.5-2.0).

Return JSON only, no Markdown:
{{
    "agent_configs": [
        {{
            "agent_id": <must match the input>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <posting frequency>,
            "comments_per_hour": <commenting frequency>,
            "active_hours": [<active hour list>],
            "response_delay_min": <minimum response delay in minutes>,
            "response_delay_max": <maximum response delay in minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influence weight>
        }},
        ...
    ]
}}"""

        system_prompt = "You are a social media behavior analysis expert. Return pure JSON. Configuration must fit the target audience's likely daily rhythm."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'stance' field value MUST be one of the English strings: 'supportive', 'opposing', 'neutral', 'observer'. All JSON field names and numeric values must remain unchanged. Only natural language text fields should use the specified language."

        return prompt, system_prompt

    def _agent_configs_from_llm_configs(
        self,
        entities: List[EntityNode],
        start_idx: int,
        llm_configs: Dict[int, Dict[str, Any]],
    ) -> List[AgentActivityConfig]:
        """Build AgentActivityConfig objects from LLM output with rule fallbacks."""
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)

        return configs

    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """Generate one batch of Agent configuration."""
        prompt, system_prompt = self._build_agent_config_batch_prompt(
            entities,
            start_idx,
            simulation_requirement,
        )

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning("Agent config batch LLM generation failed: %s; using rules", e)
            llm_configs = {}

        return self._agent_configs_from_llm_configs(entities, start_idx, llm_configs)

    async def _generate_agent_configs_batch_async(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """Generate one Agent configuration batch with async LLM I/O."""
        prompt, system_prompt = self._build_agent_config_batch_prompt(
            entities,
            start_idx,
            simulation_requirement,
        )

        try:
            result = await self._call_llm_with_retry_async(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning("Async Agent config batch LLM generation failed: %s; using rules", e)
            llm_configs = {}

        return self._agent_configs_from_llm_configs(entities, start_idx, llm_configs)

    async def _generate_agent_config_batches_async(
        self,
        context: str,
        entities: List[EntityNode],
        simulation_requirement: str,
        num_batches: int,
        max_workers: int,
        report_progress: Callable[[int, str], None],
    ) -> List[List[AgentActivityConfig]]:
        """Generate all Agent config batches with bounded async concurrency."""
        import asyncio

        semaphore = asyncio.Semaphore(max(1, max_workers))
        all_agent_configs_by_batch: List[List[AgentActivityConfig]] = [[] for _ in range(num_batches)]

        async def run_batch(batch_idx: int):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]
            async with semaphore:
                configs = await self._generate_agent_configs_batch_async(
                    context,
                    batch_entities,
                    start_idx,
                    simulation_requirement,
                )
            return batch_idx, start_idx, end_idx, configs

        tasks = [asyncio.create_task(run_batch(batch_idx)) for batch_idx in range(num_batches)]
        for task in asyncio.as_completed(tasks):
            batch_idx, start_idx, end_idx, configs = await task
            report_progress(
                3 + batch_idx,
                t('progress.generatingAgentConfig', start=start_idx + 1, end=end_idx, total=len(entities))
            )
            all_agent_configs_by_batch[batch_idx] = configs

        return all_agent_configs_by_batch
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Generate one Agent configuration with rule-based defaults."""
        entity_type = (entity.get_entity_type() or "Unknown").lower()
        
        if entity_type in ["university", "governmentagency", "ngo"]:
            # Official institutions: work-hour activity, low frequency, high influence.
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            # Media: broad daily activity, moderate frequency, high influence.
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # Experts/professors: workday plus evening activity, moderate frequency.
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # Students: mostly evening activity, high frequency.
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            # Alumni: mostly evening activity.
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # General participants: evening peak.
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    
