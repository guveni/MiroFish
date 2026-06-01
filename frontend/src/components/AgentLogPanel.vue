<template>
  <div class="right-panel">
    <div class="panel-header" :class="`panel-header--${activeStep.status}`" v-if="!isComplete">
      <span class="header-dot" v-if="activeStep.status === 'active'"></span>
      <span class="header-index mono">{{ activeStep.noLabel }}</span>
      <span class="header-title">{{ activeStep.title }}</span>
      <span class="header-meta mono" v-if="activeStep.meta">{{ activeStep.meta }}</span>
    </div>

    <!-- Workflow Overview -->
    <div class="workflow-overview" v-if="agentLogs.length > 0">
      <div class="workflow-metrics">
        <div class="metric">
          <span class="metric-label">Sections</span>
          <span class="metric-value mono">{{ completedSections }}/{{ totalSections }}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Elapsed</span>
          <span class="metric-value mono">{{ formatElapsedTime }}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Tools</span>
          <span class="metric-value mono">{{ totalToolCalls }}</span>
        </div>
        <div class="metric metric-right">
          <span class="metric-pill" :class="`pill--${statusClass}`">{{ statusText }}</span>
        </div>
      </div>

      <div class="workflow-steps" v-if="workflowSteps.length > 0">
        <div
          v-for="(step, sidx) in workflowSteps"
          :key="step.key"
          class="wf-step"
          :class="`wf-step--${step.status}`"
        >
          <div class="wf-step-connector">
            <div class="wf-step-dot"></div>
            <div class="wf-step-line" v-if="sidx < workflowSteps.length - 1"></div>
          </div>

          <div class="wf-step-content">
            <div class="wf-step-title-row">
              <span class="wf-step-index mono">{{ step.noLabel }}</span>
              <span class="wf-step-title">{{ step.title }}</span>
              <span class="wf-step-meta mono" v-if="step.meta">{{ step.meta }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Next Step Button - Displayed after completion -->
      <button v-if="isComplete" class="next-step-btn" @click="$emit('go-to-interaction')">
        <span>{{ $t('step4.goToInteraction') }}</span>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="5" y1="12" x2="19" y2="12"></line>
          <polyline points="12 5 19 12 12 19"></polyline>
        </svg>
      </button>

      <!-- Continue/Retry Button - Displayed on failure -->
      <button v-if="isFailed" class="next-step-btn retry-btn" @click="$emit('handle-continue')" :disabled="isContinuing">
        <span>{{ isContinuing ? $t('step4.continuing') : $t('step4.continueGeneration') }}</span>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" class="spin-icon" v-if="isContinuing">
          <circle cx="12" cy="12" r="10" stroke-width="4" stroke="#E5E7EB"></circle>
          <path d="M12 2a10 10 0 0 1 10 10" stroke-width="4" stroke="#4B5563" stroke-linecap="round"></path>
        </svg>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" v-else>
          <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path>
        </svg>
      </button>

      <div class="workflow-divider"></div>
    </div>

    <div class="workflow-timeline">
      <TransitionGroup name="timeline-item">
        <div 
          v-for="(log, idx) in displayLogs" 
          :key="log.timestamp + '-' + idx"
          class="timeline-item"
          :class="getTimelineItemClass(log, idx, displayLogs.length)"
        >
          <!-- Timeline Connector -->
          <div class="timeline-connector">
            <div class="connector-dot" :class="getConnectorClass(log, idx, displayLogs.length)"></div>
            <div class="connector-line" v-if="idx < displayLogs.length - 1"></div>
          </div>
          
          <!-- Timeline Content -->
          <div class="timeline-content">
            <div class="timeline-header">
              <span class="action-label">{{ getActionLabel(log.action) }}</span>
              <span class="action-time">{{ formatTime(log.timestamp) }}</span>
            </div>
            
            <!-- Action Body - Different for each type -->
            <div class="timeline-body" :class="{ 'collapsed': isLogCollapsed(log) }" @click="$emit('toggle-log-expand', log)">
              
              <!-- Report Start -->
              <template v-if="log.action === 'report_start'">
                <div class="info-row">
                  <span class="info-key">Simulation</span>
                  <span class="info-val mono">{{ log.details?.simulation_id }}</span>
                </div>
                <div class="info-row" v-if="log.details?.simulation_requirement">
                  <span class="info-key">Requirement</span>
                  <span class="info-val">{{ log.details.simulation_requirement }}</span>
                </div>
              </template>

              <!-- Planning -->
              <template v-if="log.action === 'planning_start'">
                <div class="status-message planning">{{ log.details?.message }}</div>
              </template>
              <template v-if="log.action === 'planning_complete'">
                <div class="status-message success">{{ log.details?.message }}</div>
                <div class="outline-badge" v-if="log.details?.outline">
                  {{ log.details.outline.sections?.length || 0 }} sections planned
                </div>
              </template>

              <!-- Section Start -->
              <template v-if="log.action === 'section_start'">
                <div class="section-tag">
                  <span class="tag-num">#{{ log.section_index }}</span>
                  <span class="tag-title">{{ log.section_title }}</span>
                </div>
              </template>
              
              <!-- Section Content Generated -->
              <template v-if="log.action === 'section_content'">
                <div class="section-tag content-ready">
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 20h9"></path>
                    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                  </svg>
                  <span class="tag-title">{{ log.section_title }}</span>
                </div>
              </template>

              <!-- Section Complete -->
              <template v-if="log.action === 'section_complete'">
                <div class="section-tag completed">
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                  <span class="tag-title">{{ log.section_title }}</span>
                </div>
              </template>

              <!-- Tool Call -->
              <template v-if="log.action === 'tool_call'">
                <div class="tool-badge" :class="'tool-' + getToolColor(log.details?.tool_name)">
                  <svg v-if="getToolIcon(log.details?.tool_name) === 'lightbulb'" class="tool-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.5V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.5A7 7 0 0 0 12 2z"></path>
                  </svg>
                  <svg v-else-if="getToolIcon(log.details?.tool_name) === 'globe'" class="tool-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                  </svg>
                  <svg v-else-if="getToolIcon(log.details?.tool_name) === 'users'" class="tool-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                    <circle cx="9" cy="7" r="4"></circle>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"></path>
                  </svg>
                  <svg v-else-if="getToolIcon(log.details?.tool_name) === 'zap'" class="tool-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                  </svg>
                  <svg v-else-if="getToolIcon(log.details?.tool_name) === 'chart'" class="tool-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                  </svg>
                  <svg v-else-if="getToolIcon(log.details?.tool_name) === 'database'" class="tool-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
                    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
                  </svg>
                  <svg v-else class="tool-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
                  </svg>
                  {{ getToolDisplayName(log.details?.tool_name) }}
                </div>
                <div v-if="log.details?.parameters && expandedLogs.has(log.timestamp)" class="tool-params">
                  <pre>{{ formatParams(log.details.parameters) }}</pre>
                </div>
              </template>

              <!-- Tool Result -->
              <template v-if="log.action === 'tool_result'">
                <div class="result-wrapper" :class="'result-' + log.details?.tool_name">
                  <div v-if="!['interview_agents', 'insight_forge', 'panorama_search', 'quick_search'].includes(log.details?.tool_name)" class="result-meta">
                    <span class="result-tool">{{ getToolDisplayName(log.details?.tool_name) }}</span>
                    <span class="result-size">{{ formatResultSize(log.details?.result_length) }}</span>
                  </div>
                  
                  <!-- Structured Result Display -->
                  <div v-if="!showRawResult[log.timestamp]" class="result-structured">
                    <!-- Interview Agents -->
                    <template v-if="log.details?.tool_name === 'interview_agents'">
                      <InterviewDisplay :result="parseInterviewAgents(log.details.result)" :result-length="log.details?.result_length" />
                    </template>
                    
                    <!-- Insight Forge -->
                    <template v-else-if="log.details?.tool_name === 'insight_forge'">
                      <InsightDisplay :result="parseInsightForge(log.details.result)" :result-length="log.details?.result_length" />
                    </template>
                    
                    <!-- Panorama Search -->
                    <template v-else-if="log.details?.tool_name === 'panorama_search'">
                      <PanoramaDisplay :result="parsePanorama(log.details.result)" :result-length="log.details?.result_length" />
                    </template>
                    
                    <!-- Quick Search -->
                    <template v-else-if="log.details?.tool_name === 'quick_search'">
                      <QuickSearchDisplay :result="parseQuickSearch(log.details.result)" :result-length="log.details?.result_length" />
                    </template>
                    
                    <!-- Default -->
                    <template v-else>
                      <pre class="raw-preview">{{ truncateText(log.details?.result, 300) }}</pre>
                    </template>
                  </div>
                  
                  <!-- Raw Result -->
                  <div v-else class="result-raw">
                    <pre>{{ log.details?.result }}</pre>
                  </div>
                  
                  <!-- Toggle Raw / Structured View Button -->
                  <button class="toggle-raw-btn" @click.stop="$emit('toggle-raw-result', log.timestamp, $event)">
                    <span>{{ showRawResult[log.timestamp] ? $t('step4.structuredView') : $t('step4.rawResponse') }}</span>
                  </button>
                </div>
              </template>

              <!-- LLM Response -->
              <template v-if="log.action === 'llm_response'">
                <div class="llm-response-wrapper">
                  <div class="response-header">
                    <span class="response-label">LLM RESPONSE</span>
                    <span class="response-size">{{ formatResultSize(log.details?.response_length) }}</span>
                  </div>
                  <pre class="raw-preview">{{ truncateText(log.details?.response, 300) }}</pre>
                </div>
              </template>

              <!-- Error -->
              <template v-if="log.action === 'error'">
                <div class="status-message error">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                  <span>{{ log.details?.error }}</span>
                </div>
              </template>
              
              <!-- Report Complete -->
              <template v-if="log.action === 'report_complete'">
                <div class="status-message success">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                  </svg>
                  <span>{{ log.details?.message }}</span>
                </div>
                <div class="completed-summary" v-if="log.details">
                  <div class="summary-row">
                    <span class="summary-key">Total Sections</span>
                    <span class="summary-val mono">{{ log.details.total_sections }}</span>
                  </div>
                  <div class="summary-row">
                    <span class="summary-key">Execution Time</span>
                    <span class="summary-val mono">{{ formatDuration(log.details.total_time_seconds) }}</span>
                  </div>
                </div>
              </template>

            </div>
          </div>
        </div>
      </TransitionGroup>
    </div>

    <!-- Collapsible Bottom Console (Console Log) -->
    <div class="console-panel" :class="{ 'is-collapsed': isConsoleCollapsed }">
      <div class="console-header" @click="$emit('toggle-console-collapse')">
        <div class="header-left">
          <span class="header-indicator"></span>
          <span class="header-title">CONSOLE LOG</span>
          <span class="header-count mono">{{ consoleLogs.length }} lines</span>
        </div>
        <svg class="collapse-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="18 15 12 9 6 15" v-if="!isConsoleCollapsed"></polyline>
          <polyline points="6 9 12 15 18 9" v-else></polyline>
        </svg>
      </div>
      <div class="console-content" ref="logContent" v-show="!isConsoleCollapsed">
        <div 
          v-for="(log, idx) in consoleLogs" 
          :key="idx" 
          class="console-line" 
          :class="getConsoleLineClass(log)"
        >
          {{ log }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { h, ref, reactive } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps({
  agentLogs: Array,
  displayLogs: Array,
  expandedLogs: Object,
  showRawResult: Object,
  activeStep: Object,
  isComplete: Boolean,
  isFailed: Boolean,
  isContinuing: Boolean,
  completedSections: Number,
  totalSections: Number,
  formatElapsedTime: String,
  totalToolCalls: Number,
  statusClass: String,
  statusText: String,
  workflowSteps: Array,
  consoleLogs: Array,
  isConsoleCollapsed: Boolean,
  
  // Custom injectors (passed from useAgentLogs)
  getToolDisplayName: Function,
  getToolColor: Function,
  getToolIcon: Function,
  parseInsightForge: Function,
  parsePanorama: Function,
  parseQuickSearch: Function,
  parseInterviewAgents: Function
})

defineEmits([
  'go-to-interaction',
  'handle-continue',
  'toggle-log-expand',
  'toggle-raw-result',
  'toggle-console-collapse'
])

const logContent = ref(null)

const isLogCollapsed = (log) => {
  if (['tool_call', 'tool_result', 'llm_response'].includes(log.action)) {
    return !props.expandedLogs.has(log.timestamp)
  }
  return false
}

const getTimelineItemClass = (log, idx, total) => {
  const classes = []
  if (idx === total - 1 && !props.isComplete) classes.push('is-latest')
  if (log.action === 'report_complete') classes.push('item--success')
  if (log.action === 'error') classes.push('item--error')
  return classes.join(' ')
}

const getConnectorClass = (log, idx, total) => {
  if (log.action === 'report_complete') return 'dot--success'
  if (log.action === 'error') return 'dot--error'
  if (idx === total - 1 && !props.isComplete) return 'dot--active'
  return ''
}

const getActionLabel = (action) => {
  const labels = {
    'report_start': 'SYSTEM START',
    'planning_start': 'PLANNING START',
    'planning_context': 'RETRIEVE CONTEXT',
    'planning_complete': 'PLANNING SUCCESS',
    'section_start': 'SECTION INITIATE',
    'react_thought': 'RE-ACT THOUGHT',
    'tool_call': 'TOOL INVOCATION',
    'tool_result': 'TOOL OBSERVATION',
    'llm_response': 'LLM RESPONSE',
    'section_content': 'SECTION SYNTHESIS',
    'section_complete': 'SECTION COMPLETED',
    'report_complete': 'REPORT ARCHIVED',
    'error': 'CRITICAL FAILURE'
  }
  return labels[action] || action.toUpperCase()
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const truncateText = (text, maxLen) => {
  if (!text) return ''
  if (text.length <= maxLen) return text
  return text.substring(0, maxLen) + '...'
}

const formatResultSize = (length) => {
  if (!length) return ''
  if (length >= 1000) {
    return `${(length / 1000).toFixed(1)}KB`
  }
  return `${length}B`
}

const formatParams = (params) => {
  try {
    return JSON.stringify(params, null, 2)
  } catch (e) {
    return params
  }
}

const formatDuration = (seconds) => {
  if (!seconds) return '0s'
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
}

const getConsoleLineClass = (line) => {
  if (line.includes('ERROR:') || line.includes('CRITICAL:')) return 'line--error'
  if (line.includes('WARNING:')) return 'line--warn'
  if (line.includes('SUCCESS:')) return 'line--success'
  return ''
}

// Format result size for sub-components
const formatSize = (length) => {
  if (!length) return ''
  if (length >= 1000) {
    return `${(length / 1000).toFixed(1)}k chars`
  }
  return `${length} chars`
}

// Check placeholder text
const isPlaceholderText = (text) => {
  if (!text) return true
  const t = text.trim()
  return t === '（该平台未获得回复）' || t === '(该平台未获得回复)' || t === '[无回复]'
}

// Special Sub-components inside Log timeline

// Insight Display Component
const InsightDisplay = {
  props: ['result', 'resultLength'],
  setup(props) {
    const activeTab = ref('facts')
    const expandedFacts = ref(false)
    const expandedEntities = ref(false)
    const expandedRelations = ref(false)
    const INITIAL_SHOW_COUNT = 5
    
    return () => h('div', { class: 'insight-display' }, [
      h('div', { class: 'insight-header' }, [
        h('div', { class: 'header-main' }, [
          h('div', { class: 'header-title' }, 'Deep Insight'),
          h('div', { class: 'header-stats' }, [
            h('span', { class: 'stat-item' }, [
              h('span', { class: 'stat-value' }, props.result.stats.facts || props.result.facts.length),
              h('span', { class: 'stat-label' }, 'Facts')
            ]),
            h('span', { class: 'stat-divider' }, '/'),
            h('span', { class: 'stat-item' }, [
              h('span', { class: 'stat-value' }, props.result.stats.entities || props.result.entities.length),
              h('span', { class: 'stat-label' }, 'Entities')
            ]),
            h('span', { class: 'stat-divider' }, '/'),
            h('span', { class: 'stat-item' }, [
              h('span', { class: 'stat-value' }, props.result.stats.relationships || props.result.relations.length),
              h('span', { class: 'stat-label' }, 'Relations')
            ]),
            props.resultLength && h('span', { class: 'stat-divider' }, '·'),
            props.resultLength && h('span', { class: 'stat-size' }, formatSize(props.resultLength))
          ])
        ]),
        props.result.query && h('div', { class: 'header-topic' }, props.result.query),
        props.result.simulationRequirement && h('div', { class: 'header-scenario' }, [
          h('span', { class: 'scenario-label' }, t('step4.scenarioLabel')),
          h('span', { class: 'scenario-text' }, props.result.simulationRequirement)
        ])
      ]),
      
      h('div', { class: 'insight-tabs' }, [
        h('button', {
          class: ['insight-tab', { active: activeTab.value === 'facts' }],
          onClick: (e) => { e.stopPropagation(); activeTab.value = 'facts' }
        }, t('step4.tabKeyFacts', { count: props.result.facts.length })),
        h('button', {
          class: ['insight-tab', { active: activeTab.value === 'entities' }],
          onClick: (e) => { e.stopPropagation(); activeTab.value = 'entities' }
        }, t('step4.tabCoreEntities', { count: props.result.entities.length })),
        h('button', {
          class: ['insight-tab', { active: activeTab.value === 'relations' }],
          onClick: (e) => { e.stopPropagation(); activeTab.value = 'relations' }
        }, t('step4.tabRelationChains', { count: props.result.relations.length })),
        props.result.subQueries.length > 0 && h('button', {
          class: ['insight-tab', { active: activeTab.value === 'subqueries' }],
          onClick: (e) => { e.stopPropagation(); activeTab.value = 'subqueries' }
        }, t('step4.tabSubQueries', { count: props.result.subQueries.length }))
      ]),
      
      h('div', { class: 'insight-content' }, [
        activeTab.value === 'facts' && props.result.facts.length > 0 && h('div', { class: 'facts-panel' }, [
          h('div', { class: 'panel-header' }, [
            h('span', { class: 'panel-title' }, t('step4.panelKeyFacts')),
            h('span', { class: 'panel-count' }, t('step4.totalCount', { count: props.result.facts.length }))
          ]),
          h('div', { class: 'facts-list' },
            (expandedFacts.value ? props.result.facts : props.result.facts.slice(0, INITIAL_SHOW_COUNT)).map((fact, i) => 
              h('div', { class: 'fact-item', key: i }, [
                h('span', { class: 'fact-number' }, i + 1),
                h('div', { class: 'fact-content' }, fact)
              ])
            )
          ),
          props.result.facts.length > INITIAL_SHOW_COUNT && h('button', {
            class: 'expand-btn',
            onClick: (e) => { e.stopPropagation(); expandedFacts.value = !expandedFacts.value }
          }, expandedFacts.value ? t('step4.collapse') : t('step4.expandAll', { count: props.result.facts.length }))
        ]),

        activeTab.value === 'entities' && props.result.entities.length > 0 && h('div', { class: 'entities-panel' }, [
          h('div', { class: 'panel-header' }, [
            h('span', { class: 'panel-title' }, t('step4.panelCoreEntities')),
            h('span', { class: 'panel-count' }, t('step4.totalEntityCount', { count: props.result.entities.length }))
          ]),
          h('div', { class: 'entities-grid' },
            (expandedEntities.value ? props.result.entities : props.result.entities.slice(0, 12)).map((entity, i) => 
              h('div', { class: 'entity-tag', key: i, title: entity.summary || '' }, [
                h('span', { class: 'entity-name' }, entity.name),
                h('span', { class: 'entity-type' }, entity.type),
                entity.relatedFactsCount > 0 && h('span', { class: 'entity-fact-count' }, t('step4.factCount', { count: entity.relatedFactsCount }))
              ])
            )
          ),
          props.result.entities.length > 12 && h('button', {
            class: 'expand-btn',
            onClick: (e) => { e.stopPropagation(); expandedEntities.value = !expandedEntities.value }
          }, expandedEntities.value ? t('step4.collapse') : t('step4.expandAllEntities', { count: props.result.entities.length }))
        ]),

        activeTab.value === 'relations' && props.result.relations.length > 0 && h('div', { class: 'relations-panel' }, [
          h('div', { class: 'panel-header' }, [
            h('span', { class: 'panel-title' }, t('step4.panelRelationChains')),
            h('span', { class: 'panel-count' }, t('step4.totalCount', { count: props.result.relations.length }))
          ]),
          h('div', { class: 'relations-list' },
            (expandedRelations.value ? props.result.relations : props.result.relations.slice(0, INITIAL_SHOW_COUNT)).map((rel, i) => 
              h('div', { class: 'relation-item', key: i }, [
                h('span', { class: 'rel-source' }, rel.source),
                h('span', { class: 'rel-arrow' }, [
                  h('span', { class: 'rel-line' }),
                  h('span', { class: 'rel-label' }, rel.relation),
                  h('span', { class: 'rel-line' })
                ]),
                h('span', { class: 'rel-target' }, rel.target)
              ])
            )
          ),
          props.result.relations.length > INITIAL_SHOW_COUNT && h('button', {
            class: 'expand-btn',
            onClick: (e) => { e.stopPropagation(); expandedRelations.value = !expandedRelations.value }
          }, expandedRelations.value ? t('step4.collapse') : t('step4.expandAll', { count: props.result.relations.length }))
        ]),

        activeTab.value === 'subqueries' && props.result.subQueries.length > 0 && h('div', { class: 'subqueries-panel' }, [
          h('div', { class: 'panel-header' }, [
            h('span', { class: 'panel-title' }, t('step4.panelSubQueries')),
            h('span', { class: 'panel-count' }, t('step4.totalEntityCount', { count: props.result.subQueries.length }))
          ]),
          h('div', { class: 'subqueries-list' },
            props.result.subQueries.map((sq, i) => 
              h('div', { class: 'subquery-item', key: i }, [
                h('span', { class: 'subquery-number' }, `Q${i + 1}`),
                h('div', { class: 'subquery-text' }, sq)
              ])
            )
          )
        ]),
        
        activeTab.value === 'facts' && props.result.facts.length === 0 && h('div', { class: 'empty-state' }, t('step4.emptyKeyFacts')),
        activeTab.value === 'entities' && props.result.entities.length === 0 && h('div', { class: 'empty-state' }, t('step4.emptyCoreEntities')),
        activeTab.value === 'relations' && props.result.relations.length === 0 && h('div', { class: 'empty-state' }, t('step4.emptyRelationChains'))
      ])
    ])
  }
}

// Panorama Display Component
const PanoramaDisplay = {
  props: ['result', 'resultLength'],
  setup(props) {
    const activeTab = ref('active')
    const expandedActive = ref(false)
    const expandedHistorical = ref(false)
    const expandedEntities = ref(false)
    const INITIAL_SHOW_COUNT = 5
    
    return () => h('div', { class: 'panorama-display' }, [
      h('div', { class: 'panorama-header' }, [
        h('div', { class: 'header-main' }, [
          h('div', { class: 'header-title' }, 'Panorama Search'),
          h('div', { class: 'header-stats' }, [
            h('span', { class: 'stat-item' }, [
              h('span', { class: 'stat-value' }, props.result.stats.nodes),
              h('span', { class: 'stat-label' }, 'Nodes')
            ]),
            h('span', { class: 'stat-divider' }, '/'),
            h('span', { class: 'stat-item' }, [
              h('span', { class: 'stat-value' }, props.result.stats.edges),
              h('span', { class: 'stat-label' }, 'Edges')
            ]),
            props.resultLength && h('span', { class: 'stat-divider' }, '·'),
            props.resultLength && h('span', { class: 'stat-size' }, formatSize(props.resultLength))
          ])
        ]),
        props.result.query && h('div', { class: 'header-topic' }, props.result.query)
      ]),
      
      h('div', { class: 'panorama-tabs' }, [
        h('button', {
          class: ['panorama-tab', { active: activeTab.value === 'active' }],
          onClick: (e) => { e.stopPropagation(); activeTab.value = 'active' }
        }, t('step4.tabActiveFacts', { count: props.result.activeFacts.length })),
        h('button', {
          class: ['panorama-tab', { active: activeTab.value === 'historical' }],
          onClick: (e) => { e.stopPropagation(); activeTab.value = 'historical' }
        }, t('step4.tabHistoricalFacts', { count: props.result.historicalFacts.length })),
        h('button', {
          class: ['panorama-tab', { active: activeTab.value === 'entities' }],
          onClick: (e) => { e.stopPropagation(); activeTab.value = 'entities' }
        }, t('step4.tabEntities', { count: props.result.entities.length }))
      ]),
      
      h('div', { class: 'panorama-content' }, [
        activeTab.value === 'active' && h('div', { class: 'facts-panel active-facts' }, [
          h('div', { class: 'panel-header' }, [
            h('span', { class: 'panel-title' }, t('step4.panelActiveFacts')),
            h('span', { class: 'panel-count' }, t('step4.totalCount', { count: props.result.activeFacts.length }))
          ]),
          props.result.activeFacts.length > 0 ? h('div', { class: 'facts-list' },
            (expandedActive.value ? props.result.activeFacts : props.result.activeFacts.slice(0, INITIAL_SHOW_COUNT)).map((fact, i) => 
              h('div', { class: 'fact-item active', key: i }, [
                h('span', { class: 'fact-number' }, i + 1),
                h('div', { class: 'fact-content' }, fact)
              ])
            )
          ) : h('div', { class: 'empty-state' }, t('step4.emptyActiveFacts')),
          props.result.activeFacts.length > INITIAL_SHOW_COUNT && h('button', {
            class: 'expand-btn',
            onClick: (e) => { e.stopPropagation(); expandedActive.value = !expandedActive.value }
          }, expandedActive.value ? t('step4.collapse') : t('step4.expandAll', { count: props.result.activeFacts.length }))
        ]),
        
        activeTab.value === 'historical' && h('div', { class: 'facts-panel historical-facts' }, [
          h('div', { class: 'panel-header' }, [
            h('span', { class: 'panel-title' }, t('step4.panelHistoricalFacts')),
            h('span', { class: 'panel-count' }, t('step4.totalCount', { count: props.result.historicalFacts.length }))
          ]),
          props.result.historicalFacts.length > 0 ? h('div', { class: 'facts-list' },
            (expandedHistorical.value ? props.result.historicalFacts : props.result.historicalFacts.slice(0, INITIAL_SHOW_COUNT)).map((fact, i) => 
              h('div', { class: 'fact-item historical', key: i }, [
                h('span', { class: 'fact-number' }, i + 1),
                h('div', { class: 'fact-content' }, [
                  (() => {
                    const timeMatch = fact.match(/^\[(.+?)\]\s*(.*)$/)
                    if (timeMatch) {
                      return [
                        h('span', { class: 'fact-time' }, timeMatch[1]),
                        h('span', { class: 'fact-text' }, timeMatch[2])
                      ]
                    }
                    return h('span', { class: 'fact-text' }, fact)
                  })()
                ])
              ])
            )
          ) : h('div', { class: 'empty-state' }, t('step4.emptyHistoricalFacts')),
          props.result.historicalFacts.length > INITIAL_SHOW_COUNT && h('button', {
            class: 'expand-btn',
            onClick: (e) => { e.stopPropagation(); expandedHistorical.value = !expandedHistorical.value }
          }, expandedHistorical.value ? t('step4.collapse') : t('step4.expandAll', { count: props.result.historicalFacts.length }))
        ]),
        
        activeTab.value === 'entities' && h('div', { class: 'entities-panel' }, [
          h('div', { class: 'panel-header' }, [
            h('span', { class: 'panel-title' }, t('step4.panelEntities')),
            h('span', { class: 'panel-count' }, t('step4.totalEntityCount', { count: props.result.entities.length }))
          ]),
          props.result.entities.length > 0 ? h('div', { class: 'entities-grid' },
            (expandedEntities.value ? props.result.entities : props.result.entities.slice(0, 8)).map((entity, i) => 
              h('div', { class: 'entity-tag', key: i }, [
                h('span', { class: 'entity-name' }, entity.name),
                entity.type && h('span', { class: 'entity-type' }, entity.type)
              ])
            )
          ) : h('div', { class: 'empty-state' }, t('step4.emptyEntities')),
          props.result.entities.length > 8 && h('button', {
            class: 'expand-btn',
            onClick: (e) => { e.stopPropagation(); expandedEntities.value = !expandedEntities.value }
          }, expandedEntities.value ? t('step4.collapse') : t('step4.expandAllEntities', { count: props.result.entities.length }))
        ])
      ])
    ])
  }
}

// Quick Search Display Component
const QuickSearchDisplay = {
  props: ['result', 'resultLength'],
  setup(props) {
    const activeTab = ref('facts')
    const expandedFacts = ref(false)
    const INITIAL_SHOW_COUNT = 5
    
    return () => h('div', { class: 'quick-search-display' }, [
      h('div', { class: 'quick-search-header' }, [
        h('div', { class: 'header-main' }, [
          h('div', { class: 'header-title' }, 'Quick Search'),
          h('div', { class: 'header-stats' }, [
            h('span', { class: 'stat-item' }, [
              h('span', { class: 'stat-value' }, props.result.count),
              h('span', { class: 'stat-label' }, 'Results')
            ]),
            props.resultLength && h('span', { class: 'stat-divider' }, '·'),
            props.resultLength && h('span', { class: 'stat-size' }, formatSize(props.resultLength))
          ])
        ]),
        props.result.query && h('div', { class: 'header-topic' }, props.result.query)
      ]),
      
      h('div', { class: 'quick-search-content' }, [
        h('div', { class: 'facts-panel' }, [
          props.result.facts.length > 0 ? h('div', { class: 'facts-list' },
            (expandedFacts.value ? props.result.facts : props.result.facts.slice(0, INITIAL_SHOW_COUNT)).map((fact, i) => 
              h('div', { class: 'fact-item', key: i }, [
                h('span', { class: 'fact-number' }, i + 1),
                h('div', { class: 'fact-content' }, fact)
              ])
            )
          ) : h('div', { class: 'empty-state' }, t('step4.emptyKeyFacts')),
          props.result.facts.length > INITIAL_SHOW_COUNT && h('button', {
            class: 'expand-btn',
            onClick: (e) => { e.stopPropagation(); expandedFacts.value = !expandedFacts.value }
          }, expandedFacts.value ? t('step4.collapse') : t('step4.expandAll', { count: props.result.facts.length }))
        ])
      ])
    ])
  }
}

// Interview Agents Display Component
const InterviewDisplay = {
  props: ['result', 'resultLength'],
  setup(props) {
    const activeIndex = ref(0)
    const expandedAnswers = ref(new Set())
    const platformTabs = reactive({})
    
    const getPlatformTab = (agentIdx, qIdx) => {
      const key = `${agentIdx}-${qIdx}`
      return platformTabs[key] || 'twitter'
    }
    
    const setPlatformTab = (agentIdx, qIdx, platform) => {
      const key = `${agentIdx}-${qIdx}`
      platformTabs[key] = platform
    }
    
    const toggleAnswer = (key) => {
      const newSet = new Set(expandedAnswers.value)
      if (newSet.has(key)) {
        newSet.delete(key)
      } else {
        newSet.add(key)
      }
      expandedAnswers.value = newSet
    }
    
    const formatAnswer = (text, expanded) => {
      if (!text) return ''
      if (expanded || text.length <= 400) return text
      return text.substring(0, 400) + '...'
    }

    const splitAnswerByQuestions = (answerText, questionCount) => {
      if (!answerText || questionCount <= 0) return [answerText]
      if (isPlaceholderText(answerText)) return ['']

      let matches = []
      let match

      const cnPattern = /(?:^|[\r\n]+)问题(\d+)[：:]\s*/g
      while ((match = cnPattern.exec(answerText)) !== null) {
        matches.push({
          num: parseInt(match[1]),
          index: match.index,
          fullMatch: match[0]
        })
      }

      if (matches.length === 0) {
        const numPattern = /(?:^|[\r\n]+)(\d+)\.\s+/g
        while ((match = numPattern.exec(answerText)) !== null) {
          matches.push({
            num: parseInt(match[1]),
            index: match.index,
            fullMatch: match[0]
          })
        }
      }

      if (matches.length <= 1) {
        const cleaned = answerText
          .replace(/^问题\d+[：:]\s*/, '')
          .replace(/^\d+\.\s+/, '')
          .trim()
        return [cleaned || answerText]
      }

      const parts = []
      for (let i = 0; i < matches.length; i++) {
        const current = matches[i]
        const next = matches[i + 1]

        const startIdx = current.index + current.fullMatch.length
        const endIdx = next ? next.index : answerText.length

        let part = answerText.substring(startIdx, endIdx).trim()
        part = part.replace(/[\r\n]+$/, '').trim()
        parts.push(part)
      }

      if (parts.length > 0 && parts.some(p => p)) {
        return parts
      }

      return [answerText]
    }
    
    const getAnswerForQuestion = (interview, qIdx, platform) => {
      const answer = platform === 'twitter' ? interview.twitterAnswer : (interview.redditAnswer || interview.twitterAnswer)
      if (!answer || isPlaceholderText(answer)) return answer || ''

      const questionCount = interview.questions?.length || 1
      const answers = splitAnswerByQuestions(answer, questionCount)

      if (answers.length > 1 && qIdx < answers.length) {
        return answers[qIdx] || ''
      }

      return qIdx === 0 ? answer : ''
    }
    
    const hasMultiplePlatforms = (interview, qIdx) => {
      if (!interview.twitterAnswer || !interview.redditAnswer) return false
      const twitterAnswer = getAnswerForQuestion(interview, qIdx, 'twitter')
      const redditAnswer = getAnswerForQuestion(interview, qIdx, 'reddit')
      return !isPlaceholderText(twitterAnswer) && !isPlaceholderText(redditAnswer) && twitterAnswer !== redditAnswer
    }
    
    return () => h('div', { class: 'interview-display' }, [
      h('div', { class: 'interview-header' }, [
        h('div', { class: 'header-main' }, [
          h('div', { class: 'header-title' }, 'Agent Interview'),
          h('div', { class: 'header-stats' }, [
            h('span', { class: 'stat-item' }, [
              h('span', { class: 'stat-value' }, props.result.successCount || props.result.interviews.length),
              h('span', { class: 'stat-label' }, 'Interviewed')
            ]),
            props.result.totalCount > 0 && h('span', { class: 'stat-divider' }, '/'),
            props.result.totalCount > 0 && h('span', { class: 'stat-item' }, [
              h('span', { class: 'stat-value' }, props.result.totalCount),
              h('span', { class: 'stat-label' }, 'Total')
            ]),
            props.resultLength && h('span', { class: 'stat-divider' }, '·'),
            props.resultLength && h('span', { class: 'stat-size' }, formatSize(props.resultLength))
          ])
        ]),
        props.result.topic && h('div', { class: 'header-topic' }, props.result.topic)
      ]),
      
      props.result.interviews.length > 0 && h('div', { class: 'agent-tabs' }, 
        props.result.interviews.map((interview, i) => h('button', {
          class: ['agent-tab', { active: activeIndex.value === i }],
          key: i,
          onClick: (e) => { e.stopPropagation(); activeIndex.value = i }
        }, [
          h('span', { class: 'tab-avatar' }, interview.name ? interview.name.charAt(0) : (i + 1)),
          h('span', { class: 'tab-name' }, interview.title || interview.name || `Agent ${i + 1}`)
        ]))
      ),
      
      props.result.interviews.length > 0 && h('div', { class: 'interview-detail' }, [
        h('div', { class: 'agent-profile' }, [
          h('div', { class: 'profile-avatar' }, props.result.interviews[activeIndex.value]?.name?.charAt(0) || 'A'),
          h('div', { class: 'profile-info' }, [
            h('div', { class: 'profile-name' }, props.result.interviews[activeIndex.value]?.name || 'Agent'),
            h('div', { class: 'profile-role' }, props.result.interviews[activeIndex.value]?.role || ''),
            props.result.interviews[activeIndex.value]?.bio && h('div', { class: 'profile-bio' }, props.result.interviews[activeIndex.value].bio)
          ])
        ]),
        
        props.result.interviews[activeIndex.value]?.selectionReason && h('div', { class: 'selection-reason' }, [
          h('div', { class: 'reason-label' }, 'Selection Reason'),
          h('div', { class: 'reason-content' }, props.result.interviews[activeIndex.value].selectionReason)
        ]),
        
        h('div', { class: 'qa-thread' }, 
          (props.result.interviews[activeIndex.value]?.questions?.length > 0 
            ? props.result.interviews[activeIndex.value].questions 
            : [props.result.interviews[activeIndex.value]?.question || 'No question available']
          ).map((question, qIdx) => {
            const interview = props.result.interviews[activeIndex.value]
            const currentPlatform = getPlatformTab(activeIndex.value, qIdx)
            const answerText = getAnswerForQuestion(interview, qIdx, currentPlatform)
            const hasDualPlatform = hasMultiplePlatforms(interview, qIdx)
            const expandKey = `${activeIndex.value}-${qIdx}`
            const isExpanded = expandedAnswers.value.has(expandKey)
            const isPlaceholder = isPlaceholderText(answerText)

            return h('div', { class: 'qa-pair', key: qIdx }, [
              h('div', { class: 'qa-question' }, [
                h('div', { class: 'qa-badge q-badge' }, `Q${qIdx + 1}`),
                h('div', { class: 'qa-content' }, [
                  h('div', { class: 'qa-sender' }, 'Interviewer'),
                  h('div', { class: 'qa-text' }, question)
                ])
              ]),

              answerText && h('div', { class: ['qa-answer', { 'answer-placeholder': isPlaceholder }] }, [
                h('div', { class: 'qa-badge a-badge' }, `A${qIdx + 1}`),
                h('div', { class: 'qa-content' }, [
                  h('div', { class: 'qa-answer-header' }, [
                    h('div', { class: 'qa-sender' }, interview?.name || 'Agent'),
                    hasDualPlatform && h('div', { class: 'platform-switch' }, [
                      h('button', {
                        class: ['platform-btn', { active: currentPlatform === 'twitter' }],
                        onClick: (e) => { e.stopPropagation(); setPlatformTab(activeIndex.value, qIdx, 'twitter') }
                      }, [
                        h('svg', { class: 'platform-icon', viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
                          h('circle', { cx: '12', cy: '12', r: '10' }),
                          h('line', { x1: '2', y1: '12', x2: '22', y2: '12' }),
                          h('path', { d: 'M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z' })
                        ]),
                        h('span', {}, t('step4.world1'))
                      ]),
                      h('button', {
                        class: ['platform-btn', { active: currentPlatform === 'reddit' }],
                        onClick: (e) => { e.stopPropagation(); setPlatformTab(activeIndex.value, qIdx, 'reddit') }
                      }, [
                        h('svg', { class: 'platform-icon', viewBox: '0 0 24 24', width: 12, height: 12, fill: 'none', stroke: 'currentColor', 'stroke-width': 2 }, [
                          h('circle', { cx: '12', cy: '12', r: '10' }),
                          h('line', { x1: '2', y1: '12', x2: '22', y2: '12' }),
                          h('path', { d: 'M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z' })
                        ]),
                        h('span', {}, t('step4.world2'))
                      ])
                    ])
                  ]),
                  
                  h('div', { class: 'qa-text-body' }, [
                    h('div', { class: 'qa-text' }, formatAnswer(answerText, isExpanded)),
                    answerText.length > 400 && h('button', {
                      class: 'expand-text-btn',
                      onClick: (e) => { e.stopPropagation(); toggleAnswer(expandKey) }
                    }, isExpanded ? t('step4.collapse') : t('step4.readMore'))
                  ])
                ])
              ])
            ])
          })
        )
      ])
    ])
  }
}
</script>
