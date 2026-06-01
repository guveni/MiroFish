<template>
  <div class="report-panel">
    <!-- Main Split Layout -->
    <div class="main-split-layout" :class="{ 'is-focus-mode': isFocusMode }">
      <!-- LEFT PANEL: Report Style -->
      <div class="left-panel report-style" ref="leftPanel">
        <div v-if="reportOutline" class="report-content-wrapper">
          <!-- Report Header Controls -->
          <ReportHeaderControls
            :report-id="reportId"
            :is-focus-mode="isFocusMode"
            :is-recreating="isRecreating"
            :simulation-id="simulationId"
            @toggle-focus-mode="toggleFocusMode"
            @download-pdf="downloadPDF"
            @recreate-report="openRegenerateModal"
          />

          <h1 class="main-title">{{ reportOutline.title }}</h1>
          <p class="sub-title">{{ reportOutline.summary }}</p>
          <div class="header-divider"></div>

          <!-- Sections List Renderer -->
          <ReportSectionRenderer
            :report-outline="reportOutline"
            :current-section-index="currentSectionIndex"
            :generated-sections="generatedSections"
            :collapsed-sections="collapsedSections"
            @toggle-section-collapse="toggleSectionCollapse"
          />
        </div>

        <!-- Waiting State -->
        <div v-if="!reportOutline" class="waiting-placeholder">
          <div class="waiting-animation">
            <div class="waiting-ring"></div>
            <div class="waiting-ring"></div>
            <div class="waiting-ring"></div>
          </div>
          <span class="waiting-text">Waiting for Report Agent...</span>
        </div>
      </div>

      <!-- RIGHT PANEL: Workflow Timeline / Agent Log Panel -->
      <AgentLogPanel
        v-if="agentLogs.length > 0 || reportOutline"
        :agent-logs="agentLogs"
        :display-logs="displayLogs"
        :expanded-logs="expandedLogs"
        :show-raw-result="showRawResult"
        :active-step="activeStep"
        :is-complete="isComplete"
        :is-failed="isFailed"
        :is-continuing="isContinuing"
        :completed-sections="completedSections"
        :total-sections="totalSections"
        :format-elapsed-time="formatElapsedTime"
        :total-tool-calls="totalToolCalls"
        :status-class="statusClass"
        :status-text="statusText"
        :workflow-steps="workflowSteps"
        :console-logs="consoleLogs"
        :is-console-collapsed="isConsoleCollapsed"
        
        :get-tool-display-name="getToolDisplayName"
        :get-tool-color="getToolColor"
        :get-tool-icon="getToolIcon"
        :parse-insight-forge="parseInsightForge"
        :parse-panorama="parsePanorama"
        :parse-quick-search="parseQuickSearch"
        :parse-interview-agents="parseInterviewAgents"
        
        @go-to-interaction="goToInteraction"
        @handle-continue="handleContinue"
        @toggle-log-expand="toggleLogExpand"
        @toggle-raw-result="toggleRawResult"
        @toggle-console-collapse="toggleConsoleCollapse"
      />
    </div>

    <!-- Regenerate Report Modal -->
    <Transition name="modal">
      <div v-if="showRegenerateModal" class="regenerate-modal-overlay" @click.self="showRegenerateModal = false">
        <div class="regenerate-modal">
          <div class="modal-header" style="border-bottom: 1px solid #E5E7EB; padding-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0; font-size: 18px; font-weight: 600; color: #111827;">{{ $t('step4.recreateReport') || 'Regenerate Report' }}</h3>
            <button class="modal-close" @click="showRegenerateModal = false" style="background: none; border: none; font-size: 24px; color: #6B7280; cursor: pointer; line-height: 1;">&times;</button>
          </div>
          <div class="modal-body" style="padding-top: 16px;">
            <p style="margin: 0 0 16px 0; font-size: 14px; color: #4B5563; line-height: 1.5;">
              {{ $t('step4.confirmRecreate') || 'Are you sure you want to recreate this report? The current report will be deleted and regenerated from scratch. This may take several minutes.' }}
            </p>
            <div class="input-group" style="display: flex; flex-direction: column; gap: 8px;">
              <label style="font-size: 14px; font-weight: 500; color: #374151;">Optional instructions for report regeneration:</label>
              <textarea 
                v-model="regenerateInstructions" 
                placeholder="e.g. Please focus more on financial impact..." 
                rows="4"
                style="width: 100%; padding: 12px; border: 1px solid #D1D5DB; border-radius: 6px; font-family: inherit; font-size: 14px; resize: vertical; box-sizing: border-box;"
              ></textarea>
            </div>
          </div>
          <div class="modal-footer" style="margin-top: 24px; display: flex; justify-content: flex-end; gap: 12px;">
            <button 
              @click="showRegenerateModal = false"
              style="padding: 8px 16px; background: white; border: 1px solid #D1D5DB; border-radius: 6px; font-size: 14px; font-weight: 500; color: #374151; cursor: pointer;"
            >
              Cancel
            </button>
            <button 
              @click="confirmRegenerate"
              style="padding: 8px 16px; background: #3B82F6; border: none; border-radius: 6px; font-size: 14px; font-weight: 500; color: white; cursor: pointer;"
            >
              Regenerate
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { generateReport } from '../api/report'

// Sub-components
import ReportHeaderControls from './ReportHeaderControls.vue'
import ReportSectionRenderer from './ReportSectionRenderer.vue'
import AgentLogPanel from './AgentLogPanel.vue'

// Composables
import { useReportGeneration } from '../composables/useReportGeneration'
import { useAgentLogs } from '../composables/useAgentLogs'

const router = useRouter()
const { t } = useI18n()

const props = defineProps({
  reportId: String,
  simulationId: String,
  systemLogs: Array
})

const emit = defineEmits(['add-log', 'update-status'])

// Element refs
const leftPanel = ref(null)
const rightPanel = ref(null)
const logContent = ref(null)

// Focus Mode state
const isFocusMode = ref(false)
const toggleFocusMode = () => {
  isFocusMode.value = !isFocusMode.value
}

const downloadPDF = () => {
  window.print()
}

// 1. Initialize Log state
const {
  agentLogs,
  consoleLogs,
  agentLogLine,
  consoleLogLine,
  expandedLogs,
  showRawResult,
  isConsoleCollapsed,
  toggleLogExpand,
  isLogCollapsed,
  toggleRawResult,
  getToolDisplayName,
  getToolColor,
  getToolIcon,
  parseInsightForge,
  parsePanorama,
  parseQuickSearch,
  parseInterviewAgents
} = useAgentLogs()

const toggleConsoleCollapse = () => {
  isConsoleCollapsed.value = !isConsoleCollapsed.value
}

// 2. Initialize Report generation state and WebSocket polling
const onStatusUpdate = (status) => emit('update-status', status)
const addLog = (log) => emit('add-log', log)

const {
  reportOutline,
  currentSectionIndex,
  generatedSections,
  expandedContent,
  collapsedSections,
  isComplete,
  isFailed,
  isContinuing,
  isRecreating,
  startTime,
  startPolling,
  stopPolling,
  handleContinue,
  toggleSectionContent,
  toggleSectionCollapse
} = useReportGeneration({
  reportId: () => props.reportId,
  simulationId: () => props.simulationId,
  agentLogs,
  consoleLogs,
  agentLogLine,
  consoleLogLine,
  onStatusUpdate,
  addLog,
  rightPanelRef: rightPanel,
  logContentRef: logContent
})

// Navigation to Step 5
const goToInteraction = () => {
  if (props.reportId) {
    router.push({ name: 'Interaction', params: { reportId: props.reportId } })
  }
}

// Computed properties for progress tracking / step navigation
const statusClass = computed(() => {
  if (isComplete.value) return 'completed'
  if (agentLogs.value.length > 0) return 'processing'
  return 'pending'
})

const statusText = computed(() => {
  if (isComplete.value) return 'Completed'
  if (agentLogs.value.length > 0) return 'Generating...'
  return 'Waiting'
})

const totalSections = computed(() => {
  return reportOutline.value?.sections?.length || 0
})

const completedSections = computed(() => {
  return Object.keys(generatedSections.value).length
})

const progressPercent = computed(() => {
  if (totalSections.value === 0) return 0
  return Math.round((completedSections.value / totalSections.value) * 100)
})

const totalToolCalls = computed(() => {
  return agentLogs.value.filter(l => l.action === 'tool_call').length
})

const formatElapsedTime = computed(() => {
  if (!startTime.value) return '0s'
  const lastLog = agentLogs.value[agentLogs.value.length - 1]
  const elapsed = lastLog?.elapsed_seconds || 0
  if (elapsed < 60) return `${Math.round(elapsed)}s`
  const mins = Math.floor(elapsed / 60)
  const secs = Math.round(elapsed % 60)
  return `${mins}m ${secs}s`
})

const displayLogs = computed(() => {
  return agentLogs.value
})

const activeSectionIndex = computed(() => {
  if (isComplete.value) return null
  if (currentSectionIndex.value) return currentSectionIndex.value
  if (totalSections.value > 0 && completedSections.value < totalSections.value) return completedSections.value + 1
  return null
})

const isPlanningDone = computed(() => {
  return !!reportOutline.value?.sections?.length || agentLogs.value.some(l => l.action === 'planning_complete')
})

const isPlanningStarted = computed(() => {
  return agentLogs.value.some(l => l.action === 'planning_start' || l.action === 'report_start')
})

const isFinalizing = computed(() => {
  return !isComplete.value && isPlanningDone.value && totalSections.value > 0 && completedSections.value >= totalSections.value
})

const activeStep = computed(() => {
  const steps = workflowSteps.value
  const active = steps.find(s => s.status === 'active')
  if (active) return active
  
  const doneSteps = steps.filter(s => s.status === 'done')
  if (doneSteps.length > 0) return doneSteps[doneSteps.length - 1]
  
  return steps[0] || { noLabel: '--', title: 'Waiting to start', status: 'todo', meta: '' }
})

const workflowSteps = computed(() => {
  const steps = []

  // Planning / Outline
  const planningStatus = isPlanningDone.value ? 'done' : (isPlanningStarted.value ? 'active' : 'todo')
  steps.push({
    key: 'planning',
    noLabel: 'PL',
    title: 'Planning / Outline',
    status: planningStatus,
    meta: planningStatus === 'active' ? 'IN PROGRESS' : ''
  })

  // Sections (if outline exists)
  const sections = reportOutline.value?.sections || []
  sections.forEach((section, i) => {
    const idx = i + 1
    const status = (isComplete.value || !!generatedSections.value[idx])
      ? 'done'
      : (activeSectionIndex.value === idx ? 'active' : 'todo')

    steps.push({
      key: `section-${idx}`,
      noLabel: String(idx).padStart(2, '0'),
      title: section.title,
      status,
      meta: status === 'active' ? 'IN PROGRESS' : ''
    })
  })

  // Complete
  const completeStatus = isComplete.value ? 'done' : (isFinalizing.value ? 'active' : 'todo')
  steps.push({
    key: 'complete',
    noLabel: 'OK',
    title: 'Complete',
    status: completeStatus,
    meta: completeStatus === 'active' ? 'FINALIZING' : ''
  })

  return steps
})

const isSectionCompleted = (sectionIndex) => {
  return !!generatedSections.value[sectionIndex]
}

// Recreate Report Modal and Actions
const showRegenerateModal = ref(false)
const regenerateInstructions = ref('')

const openRegenerateModal = () => {
  if (isRecreating.value || !props.simulationId) return
  regenerateInstructions.value = ''
  showRegenerateModal.value = true
}

const confirmRegenerate = async () => {
  showRegenerateModal.value = false
  isRecreating.value = true
  emit('update-status', 'processing')
  
  try {
    stopPolling()
    
    const payload = {
      simulation_id: props.simulationId,
      force_regenerate: true
    }
    if (regenerateInstructions.value.trim()) {
      payload.instructions = regenerateInstructions.value.trim()
    }
    const res = await generateReport(payload)
    
    if (res.success && res.data) {
      const newReportId = res.data.report_id
      emit('add-log', `Recreating report started: ${newReportId}`)
      
      router.push({ name: 'Report', params: { reportId: newReportId }, query: { simId: props.simulationId } })
    } else {
      console.error('Failed to recreate report:', res.error)
      emit('add-log', `Recreation failed: ${res.error || 'Unknown error'}`)
      emit('update-status', 'error')
      startPolling()
    }
  } catch (err) {
    console.error('Failed to recreate report:', err)
    emit('add-log', `Recreation exception: ${err.message}`)
    emit('update-status', 'error')
    startPolling()
  } finally {
    isRecreating.value = false
  }
}

// Lifecycle and watcher bindings
onMounted(() => {
  if (props.reportId) {
    addLog(`Report Agent initialized: ${props.reportId}`)
    startPolling()
  }
})

onUnmounted(() => {
  stopPolling()
})

watch(() => props.reportId, (newId) => {
  if (newId) {
    agentLogs.value = []
    consoleLogs.value = []
    agentLogLine.value = 0
    consoleLogLine.value = 0
    reportOutline.value = null
    currentSectionIndex.value = null
    generatedSections.value = {}
    expandedContent.value = new Set()
    expandedLogs.value = new Set()
    collapsedSections.value = new Set()
    isComplete.value = false
    startTime.value = null
    
    startPolling()
  }
}, { immediate: true })

watch(isConsoleCollapsed, (newVal) => {
  if (!newVal) {
    nextTick(() => {
      if (logContent.value) {
        logContent.value.scrollTop = logContent.value.scrollHeight
      }
    })
  }
})
</script>

<style>
.report-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #F8F9FA;
  font-family: 'Inter', 'Noto Sans SC', system-ui, sans-serif;
  overflow: hidden;
}

/* Main Split Layout */
.main-split-layout {
  flex: 1;
  display: flex;
  overflow: hidden;
}

/* Panel Headers */
.panel-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 20px;
  background: #FFFFFF;
  border-bottom: 1px solid #E5E7EB;
  font-size: 13px;
  font-weight: 600;
  color: #374151;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  position: sticky;
  top: 0;
  z-index: 10;
}

.header-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #1F2937;
  box-shadow: 0 0 0 3px rgba(31, 41, 55, 0.15);
  margin-right: 10px;
  flex-shrink: 0;
  animation: pulse-dot 1.5s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%, 100% {
    box-shadow: 0 0 0 3px rgba(31, 41, 55, 0.15);
  }
  50% {
    box-shadow: 0 0 0 5px rgba(31, 41, 55, 0.1);
  }
}

.header-index {
  font-size: 12px;
  font-weight: 600;
  color: #9CA3AF;
  margin-right: 10px;
  flex-shrink: 0;
}

.header-title {
  font-size: 13px;
  font-weight: 600;
  color: #374151;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-transform: none;
  letter-spacing: 0;
}

.header-meta {
  margin-left: auto;
  font-size: 10px;
  font-weight: 600;
  color: #6B7280;
  flex-shrink: 0;
}

/* Panel header status variants */
.panel-header--active {
  background: #FAFAFA;
  border-color: #1F2937;
}

.panel-header--active .header-index {
  color: #1F2937;
}

.panel-header--active .header-title {
  color: #1F2937;
}

.panel-header--active .header-meta {
  color: #1F2937;
}

.panel-header--done {
  background: #F9FAFB;
}

.panel-header--done .header-index {
  color: #10B981;
}

.panel-header--todo .header-index,
.panel-header--todo .header-title {
  color: #9CA3AF;
}

/* Left Panel - Report Style */
.left-panel.report-style {
  width: 45%;
  min-width: 450px;
  background: #FFFFFF;
  border-right: 1px solid #E5E7EB;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  padding: 30px 50px 60px 50px;
}

.left-panel::-webkit-scrollbar {
  width: 6px;
}

.left-panel::-webkit-scrollbar-track {
  background: transparent;
}

.left-panel::-webkit-scrollbar-thumb {
  background: transparent;
  border-radius: 3px;
  transition: background 0.3s ease;
}

.left-panel:hover::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
}

.left-panel::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.25);
}

/* Report Header */
.report-content-wrapper {
  max-width: 800px;
  margin: 0 auto;
  width: 100%;
}

.report-header-block {
  margin-bottom: 30px;
}

.report-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}

.meta-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.report-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #F3F4F6;
  border: 1px solid #E5E7EB;
  color: #374151;
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-btn:hover {
  background: #E5E7EB;
  color: #111827;
}

.action-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  background: #F3F4F6;
  color: #9CA3AF;
  border-color: #E5E7EB;
}

.action-btn svg {
  flex-shrink: 0;
}

/* Focus Mode CSS */
.main-split-layout.is-focus-mode .left-panel.report-style {
  width: 100% !important;
  max-width: none !important;
  border-right: none !important;
  padding: 40px 0 80px 0 !important;
}

.main-split-layout.is-focus-mode .report-content-wrapper {
  max-width: 800px !important;
  width: 100% !important;
  margin: 0 auto !important;
  padding: 0 40px !important;
  box-sizing: border-box !important;
}

.main-split-layout.is-focus-mode .right-panel {
  display: none !important;
}

/* Printing CSS */
@media print {
  :global(html),
  :global(body),
  :global(.main-view),
  :global(.content-area),
  :global(.panel-wrapper.right) {
    height: auto !important;
    overflow: visible !important;
  }

  :global(body) * {
    visibility: hidden;
  }
  
  .report-panel,
  .main-split-layout {
    height: auto !important;
    overflow: visible !important;
  }

  .left-panel.report-style,
  .left-panel.report-style *,
  .report-content-wrapper,
  .report-content-wrapper * {
    visibility: visible;
  }
  
  .left-panel.report-style {
    position: static !important;
    left: 0;
    top: 0;
    width: 100% !important;
    max-width: none !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    overflow: visible !important;
    height: auto !important;
  }
  
  .report-content-wrapper {
    width: 100% !important;
    max-width: none !important;
    padding: 0 !important;
    margin: 0 !important;
  }
  
  .report-actions {
    display: none !important;
  }
  
  .report-header-block {
    page-break-after: avoid;
  }
  
  .report-section-item {
    page-break-inside: auto;
  }
}

.report-tag {
  background: #000000;
  color: #FFFFFF;
  font-size: 11px;
  font-weight: 700;
  padding: 4px 8px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.report-id {
  font-size: 11px;
  color: #9CA3AF;
  font-weight: 500;
  letter-spacing: 0.02em;
}

.main-title {
  font-family: 'Times New Roman', Times, serif;
  font-size: 36px;
  font-weight: 700;
  color: #111827;
  line-height: 1.2;
  margin: 0 0 16px 0;
  letter-spacing: -0.02em;
}

.sub-title {
  font-family: 'Times New Roman', Times, serif;
  font-size: 16px;
  color: #6B7280;
  font-style: italic;
  line-height: 1.6;
  margin: 0 0 30px 0;
  font-weight: 400;
}

.header-divider {
  height: 1px;
  background: #E5E7EB;
  width: 100%;
}

/* Sections List */
.sections-list {
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.report-section-item {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.section-header-row {
  display: flex;
  align-items: baseline;
  gap: 12px;
  transition: background-color 0.2s ease;
  padding: 8px 12px;
  margin: -8px -12px;
  border-radius: 8px;
}

.section-header-row.clickable {
  cursor: pointer;
}

.section-header-row.clickable:hover {
  background-color: #F9FAFB;
}

.collapse-icon {
  margin-left: auto;
  color: #9CA3AF;
  transition: transform 0.3s ease;
  flex-shrink: 0;
  align-self: center;
}

.collapse-icon.is-collapsed {
  transform: rotate(-90deg);
}

.section-number {
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px;
  color: #9CA3AF; /* Dark gray, does not change with status */
  font-weight: 500;
}

.section-title {
  font-family: 'Times New Roman', Times, serif;
  font-size: 24px;
  font-weight: 600;
  color: #111827;
  margin: 0;
  transition: color 0.3s ease;
}

/* States */
.report-section-item.is-pending .section-title {
  color: #D1D5DB;
}

.report-section-item.is-active .section-title,
.report-section-item.is-completed .section-title {
  color: #111827;
}

.section-body {
  padding-left: 28px;
  overflow: hidden;
}

/* Generated Content */
.generated-content {
  font-family: 'Inter', 'Noto Sans SC', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.8;
  color: #374151;
}

.generated-content p {
  margin-bottom: 1em;
}

.generated-content .md-h2,
.generated-content .md-h3,
.generated-content .md-h4 {
  font-family: 'Times New Roman', Times, serif;
  color: #111827;
  margin-top: 1.5em;
  margin-bottom: 0.8em;
  font-weight: 700;
}

.generated-content .md-h2 { font-size: 20px; border-bottom: 1px solid #F3F4F6; padding-bottom: 8px; }
.generated-content .md-h3 { font-size: 18px; }
.generated-content .md-h4 { font-size: 16px; }

.generated-content .md-ul,
.generated-content .md-ol {
  padding-left: 24px;
  margin: 12px 0;
}

.generated-content .md-li,
.generated-content .md-oli {
  margin: 6px 0;
}

.generated-content .md-quote {
  border-left: 3px solid #E5E7EB;
  padding-left: 16px;
  margin: 1.5em 0;
  color: #6B7280;
  font-style: italic;
  font-family: 'Times New Roman', Times, serif;
}

.generated-content .code-block {
  background: #F9FAFB;
  padding: 12px;
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  overflow-x: auto;
  margin: 1em 0;
  border: 1px solid #E5E7EB;
}

.generated-content strong {
  font-weight: 600;
  color: #111827;
}

/* Loading State */
.loading-state {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #6B7280;
  font-size: 14px;
  margin-top: 4px;
}

.loading-icon {
  width: 18px;
  height: 18px;
  animation: spin 1s linear infinite;
  display: flex;
  align-items: center;
  justify-content: center;
}

.loading-text {
  font-family: 'Times New Roman', Times, serif;
  font-size: 15px;
  color: #4B5563;
}

.cursor-blink {
  display: inline-block;
  width: 8px;
  height: 14px;
  background: #8B5CF6;
  opacity: 0.5;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  0%, 100% { opacity: 0.5; }
  50% { opacity: 0; }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Content Styles Override for this view */
.generated-content .md-h2 {
  font-family: 'Times New Roman', Times, serif;
  font-size: 18px;
  margin-top: 0;
}

/* Slide Content Transition */
.slide-content-enter-active {
  transition: opacity 0.3s ease-out;
}

.slide-content-leave-active {
  transition: opacity 0.2s ease-in;
}

.slide-content-enter-from,
.slide-content-leave-to {
  opacity: 0;
}

.slide-content-enter-to,
.slide-content-leave-from {
  opacity: 1;
}

/* Waiting Placeholder */
.waiting-placeholder {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 20px;
  padding: 40px;
  color: #9CA3AF;
}

.waiting-animation {
  position: relative;
  width: 48px;
  height: 48px;
}

.waiting-ring {
  position: absolute;
  width: 100%;
  height: 100%;
  border: 2px solid #E5E7EB;
  border-radius: 50%;
  animation: ripple 2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}

.waiting-ring:nth-child(2) {
  animation-delay: 0.4s;
}

.waiting-ring:nth-child(3) {
  animation-delay: 0.8s;
}

@keyframes ripple {
  0% { transform: scale(0.5); opacity: 1; }
  100% { transform: scale(2); opacity: 0; }
}

.waiting-text {
  font-size: 14px;
}

/* Right Panel */
.right-panel {
  flex: 1;
  background: #FFFFFF;
  overflow-y: auto;
  display: flex;
  flex-direction: column;

  /* Functional palette (low saturation, status-based) */
  --wf-border: #E5E7EB;
  --wf-divider: #F3F4F6;

  --wf-active-bg: #FAFAFA;
  --wf-active-border: #1F2937;
  --wf-active-dot: #1F2937;
  --wf-active-text: #1F2937;

  --wf-done-bg: #F9FAFB;
  --wf-done-border: #E5E7EB;
  --wf-done-dot: #10B981;

  --wf-muted-dot: #D1D5DB;
  --wf-todo-text: #9CA3AF;
}

.right-panel::-webkit-scrollbar {
  width: 6px;
}

.right-panel::-webkit-scrollbar-track {
  background: transparent;
}

.right-panel::-webkit-scrollbar-thumb {
  background: transparent;
  border-radius: 3px;
  transition: background 0.3s ease;
}

.right-panel:hover::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
}

.right-panel::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.25);
}

.mono {
  font-family: 'JetBrains Mono', monospace;
}

/* Workflow Overview */
.workflow-overview {
  padding: 16px 20px 0 20px;
}

.workflow-metrics {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.metric {
  display: inline-flex;
  align-items: baseline;
  gap: 6px;
}

.metric-right {
  margin-left: auto;
}

.metric-label {
  font-size: 11px;
  font-weight: 600;
  color: #9CA3AF;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metric-value {
  font-size: 12px;
  color: #374151;
}

.metric-pill {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--wf-border);
  background: #F9FAFB;
  color: #6B7280;
}

.metric-pill.pill--processing {
  background: var(--wf-active-bg);
  border-color: var(--wf-active-border);
  color: var(--wf-active-text);
}

.metric-pill.pill--completed {
  background: #ECFDF5;
  border-color: #A7F3D0;
  color: #065F46;
}

.metric-pill.pill--pending {
  background: transparent;
  border-style: dashed;
  color: #6B7280;
}

.workflow-steps {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-bottom: 10px;
}

.wf-step {
  display: grid;
  grid-template-columns: 24px 1fr;
  gap: 12px;
  padding: 10px 12px;
  border: 1px solid var(--wf-divider);
  border-radius: 8px;
  background: #FFFFFF;
}

.wf-step--active {
  background: var(--wf-active-bg);
  border-color: var(--wf-active-border);
}

.wf-step--done {
  background: var(--wf-done-bg);
  border-color: var(--wf-done-border);
}

.wf-step--todo {
  background: transparent;
  border-color: var(--wf-border);
  border-style: dashed;
}

.wf-step-connector {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 24px;
  flex-shrink: 0;
}

.wf-step-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--wf-muted-dot);
  border: 2px solid #FFFFFF;
  z-index: 1;
}

.wf-step-line {
  width: 2px;
  flex: 1;
  background: var(--wf-divider);
  margin-top: -2px;
}

.wf-step--active .wf-step-dot {
  background: var(--wf-active-dot);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.wf-step--done .wf-step-dot {
  background: var(--wf-done-dot);
}

.wf-step-title-row {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
}

.wf-step-index {
  font-size: 11px;
  font-weight: 700;
  color: #9CA3AF;
  letter-spacing: 0.02em;
  flex-shrink: 0;
}

.wf-step-title {
  font-family: 'Times New Roman', Times, serif;
  font-size: 13px;
  font-weight: 600;
  color: #111827;
  line-height: 1.35;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.wf-step-meta {
  margin-left: auto;
  font-size: 10px;
  font-weight: 700;
  color: var(--wf-active-text);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  flex-shrink: 0;
}

.wf-step--todo .wf-step-title,
.wf-step--todo .wf-step-index {
  color: var(--wf-todo-text);
}

.workflow-divider {
  height: 1px;
  background: var(--wf-divider);
  margin: 14px 0 0 0;
}

/* Workflow Timeline */
.workflow-timeline {
  padding: 14px 20px 24px;
  flex: 1;
}

.timeline-item {
  display: grid;
  grid-template-columns: 24px 1fr;
  gap: 12px;
  padding: 10px 12px;
  margin-bottom: 10px;
  border: 1px solid var(--wf-divider);
  border-radius: 8px;
  background: #FFFFFF;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}

.timeline-item:hover {
  background: #F9FAFB;
  border-color: var(--wf-border);
}

.timeline-item.node--active {
  background: var(--wf-active-bg);
  border-color: var(--wf-active-border);
}

.timeline-item.node--active:hover {
  background: var(--wf-active-bg);
  border-color: var(--wf-active-border);
}

.timeline-item.node--done {
  background: var(--wf-done-bg);
  border-color: var(--wf-done-border);
}

.timeline-item.node--done:hover {
  background: var(--wf-done-bg);
  border-color: var(--wf-done-border);
}

.timeline-connector {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 24px;
  flex-shrink: 0;
}

.connector-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--wf-muted-dot);
  border: 2px solid #FFFFFF;
  z-index: 1;
}

.connector-line {
  width: 2px;
  flex: 1;
  background: var(--wf-divider);
  margin-top: -2px;
}

/* Connector dot: status only */
.dot-active {
  background: var(--wf-active-dot);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.dot-done {
  background: var(--wf-done-dot);
}

.dot-muted {
  background: var(--wf-muted-dot);
}

.timeline-content {
  min-width: 0;
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
  margin: 0;
  transition: none;
}

.timeline-content:hover {
  box-shadow: none;
}

.timeline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.action-label {
  font-size: 12px;
  font-weight: 600;
  color: #374151;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.action-time {
  font-size: 11px;
  color: #9CA3AF;
  font-family: 'JetBrains Mono', monospace;
}

.timeline-body {
  font-size: 13px;
  color: #4B5563;
}

.timeline-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid #F3F4F6;
}

.elapsed-placeholder {
  flex-shrink: 0;
}

.footer-actions {
  display: flex;
  gap: 8px;
  margin-left: auto;
}

.elapsed-badge {
  font-size: 11px;
  color: #6B7280;
  background: #F3F4F6;
  padding: 2px 8px;
  border-radius: 10px;
  font-family: 'JetBrains Mono', monospace;
}

/* Timeline Body Elements */
.info-row {
  display: flex;
  gap: 8px;
  margin-bottom: 6px;
}

.info-key {
  font-size: 11px;
  color: #9CA3AF;
  min-width: 80px;
}

.info-val {
  color: #374151;
}

.status-message {
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
  border: 1px solid transparent;
}

.status-message.planning {
  background: var(--wf-active-bg);
  border-color: var(--wf-active-border);
  color: var(--wf-active-text);
}

.status-message.success {
  background: #ECFDF5;
  border-color: #A7F3D0;
  color: #065F46;
}

.outline-badge {
  display: inline-block;
  margin-top: 8px;
  padding: 4px 10px;
  background: #F9FAFB;
  color: #6B7280;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
}

.section-tag {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: #F9FAFB;
  border: 1px solid var(--wf-border);
  border-radius: 6px;
}

.section-tag.content-ready {
  background: var(--wf-active-bg);
  border: 1px dashed var(--wf-active-border);
}

.section-tag.content-ready svg {
  color: var(--wf-active-dot);
}


.section-tag.completed {
  background: #ECFDF5;
  border: 1px solid #A7F3D0;
}

.section-tag.completed svg {
  color: #059669;
}

.tag-num {
  font-size: 11px;
  font-weight: 700;
  color: #6B7280;
}

.section-tag.completed .tag-num {
  color: #059669;
}

.tag-title {
  font-size: 13px;
  font-weight: 500;
  color: #374151;
}

.tool-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: #F9FAFB;
  color: #374151;
  border: 1px solid var(--wf-border);
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  transition: all 0.2s ease;
}

.tool-icon {
  flex-shrink: 0;
}

/* Tool Colors - Purple (Deep Insight) */
.tool-badge.tool-purple {
  background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
  border-color: #C4B5FD;
  color: #6D28D9;
}
.tool-badge.tool-purple .tool-icon {
  stroke: #7C3AED;
}

/* Tool Colors - Blue (Panorama Search) */
.tool-badge.tool-blue {
  background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
  border-color: #93C5FD;
  color: #1D4ED8;
}
.tool-badge.tool-blue .tool-icon {
  stroke: #2563EB;
}

/* Tool Colors - Green (Agent Interview) */
.tool-badge.tool-green {
  background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
  border-color: #86EFAC;
  color: #15803D;
}
.tool-badge.tool-green .tool-icon {
  stroke: #16A34A;
}

/* Tool Colors - Orange (Quick Search) */
.tool-badge.tool-orange {
  background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
  border-color: #FDBA74;
  color: #C2410C;
}
.tool-badge.tool-orange .tool-icon {
  stroke: #EA580C;
}

/* Tool Colors - Cyan (Graph Stats) */
.tool-badge.tool-cyan {
  background: linear-gradient(135deg, #ECFEFF 0%, #CFFAFE 100%);
  border-color: #67E8F9;
  color: #0E7490;
}
.tool-badge.tool-cyan .tool-icon {
  stroke: #0891B2;
}

/* Tool Colors - Pink (Entity Query) */
.tool-badge.tool-pink {
  background: linear-gradient(135deg, #FDF2F8 0%, #FCE7F3 100%);
  border-color: #F9A8D4;
  color: #BE185D;
}
.tool-badge.tool-pink .tool-icon {
  stroke: #DB2777;
}

/* Tool Colors - Gray (Default) */
.tool-badge.tool-gray {
  background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
  border-color: #D1D5DB;
  color: #374151;
}
.tool-badge.tool-gray .tool-icon {
  stroke: #6B7280;
}

.tool-params {
  margin-top: 10px;
  background: transparent;
  border-radius: 0;
  padding: 10px 0 0 0;
  border-top: 1px dashed var(--wf-divider);
  overflow-x: auto;
}

.tool-params pre {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #4B5563;
  white-space: pre-wrap;
  word-break: break-all;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  padding: 10px;
}

/* Unified Action Buttons */
.action-btn {
  background: #F3F4F6;
  border: 1px solid #E5E7EB;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  color: #6B7280;
  cursor: pointer;
  transition: all 0.15s ease;
  white-space: nowrap;
}

.action-btn:hover {
  background: #E5E7EB;
  color: #374151;
  border-color: #D1D5DB;
}

.action-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  background: #F3F4F6;
  color: #9CA3AF;
  border-color: #E5E7EB;
}

/* Result Wrapper */
.result-wrapper {
  background: transparent;
  border: none;
  border-top: 1px solid var(--wf-divider);
  border-radius: 0;
  padding: 12px 0 0 0;
}

.result-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.result-tool {
  font-size: 12px;
  font-weight: 600;
  color: #374151;
}

.result-size {
  font-size: 10px;
  color: #6B7280;
  font-family: 'JetBrains Mono', monospace;
}

.result-raw {
  margin-top: 10px;
  max-height: 300px;
  overflow-y: auto;
}

.result-raw pre {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  color: #374151;
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  padding: 10px;
  border-radius: 6px;
}

.raw-preview {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  color: #6B7280;
}

/* Legacy toggle-raw removed - using unified .action-btn */

/* LLM Response */
.llm-meta {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.meta-tag {
  font-size: 11px;
  padding: 3px 8px;
  background: #F3F4F6;
  color: #6B7280;
  border-radius: 4px;
}

.meta-tag.active {
  background: #DBEAFE;
  color: #1E40AF;
}

.meta-tag.final-answer {
  background: #D1FAE5;
  color: #059669;
  font-weight: 600;
}

.final-answer-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  padding: 10px 14px;
  background: #ECFDF5;
  border: 1px solid #A7F3D0;
  border-radius: 6px;
  color: #065F46;
  font-size: 12px;
  font-weight: 500;
}

.final-answer-hint svg {
  flex-shrink: 0;
}

.llm-content {
  margin-top: 10px;
  max-height: 200px;
  overflow-y: auto;
}

.llm-content pre {
  margin: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  color: #4B5563;
  background: #F3F4F6;
  padding: 10px;
  border-radius: 6px;
}

/* Complete Banner */
.complete-banner {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  background: #ECFDF5;
  border: 1px solid #A7F3D0;
  border-radius: 8px;
  color: #065F46;
  font-weight: 600;
  font-size: 14px;
}

.next-step-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: calc(100% - 40px);
  margin: 4px 20px 0 20px;
  padding: 14px 20px;
  font-size: 14px;
  font-weight: 600;
  color: #FFFFFF;
  background: #1F2937;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.next-step-btn:hover {
  background: #374151;
}

.next-step-btn svg {
  transition: transform 0.2s ease;
}

.next-step-btn:hover svg {
  transform: translateX(4px);
}

.next-step-btn.retry-btn {
  background: #3B82F6;
}

.next-step-btn.retry-btn:hover {
  background: #2563EB;
}

.next-step-btn.retry-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.spin-icon {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Workflow Empty */
.workflow-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  color: #9CA3AF;
  font-size: 13px;
}

.empty-pulse {
  width: 24px;
  height: 24px;
  background: #E5E7EB;
  border-radius: 50%;
  margin-bottom: 16px;
  animation: pulse-ring 1.5s infinite;
}

@keyframes pulse-ring {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.5; }
}

/* Timeline Transitions */
.timeline-item-enter-active {
  transition: all 0.4s ease;
}

.timeline-item-enter-from {
  opacity: 0;
  transform: translateX(-20px);
}

/* ========== Structured Result Display Components ========== */

/* Common Styles - using :deep() for dynamic components */
:deep(.stat-row) {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

:deep(.stat-box) {
  flex: 1;
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  padding: 10px 8px;
  text-align: center;
}

:deep(.stat-box .stat-num) {
  display: block;
  font-size: 20px;
  font-weight: 700;
  color: #111827;
  font-family: 'JetBrains Mono', monospace;
}

:deep(.stat-box .stat-label) {
  display: block;
  font-size: 10px;
  color: #9CA3AF;
  margin-top: 2px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

:deep(.stat-box.highlight) {
  background: #ECFDF5;
  border-color: #A7F3D0;
}

:deep(.stat-box.highlight .stat-num) {
  color: #059669;
}

:deep(.stat-box.muted) {
  background: #F9FAFB;
  border-color: #E5E7EB;
}

:deep(.stat-box.muted .stat-num) {
  color: #6B7280;
}

:deep(.query-display) {
  background: #F9FAFB;
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 12px;
  color: #374151;
  margin-bottom: 12px;
  border: 1px solid #E5E7EB;
  line-height: 1.5;
}

:deep(.expand-details) {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 500;
  color: #6B7280;
  cursor: pointer;
  transition: all 0.15s ease;
}

:deep(.expand-details:hover) {
  border-color: #D1D5DB;
  color: #374151;
}

:deep(.detail-content) {
  margin-top: 14px;
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 14px;
}

:deep(.section-label) {
  font-size: 11px;
  font-weight: 600;
  color: #6B7280;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid #F3F4F6;
}

/* Facts Section */
:deep(.facts-section) {
  margin-bottom: 14px;
}

:deep(.fact-row) {
  display: flex;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid #F3F4F6;
}

:deep(.fact-row:last-child) {
  border-bottom: none;
}

:deep(.fact-row.active) {
  background: #ECFDF5;
  margin: 0 -10px;
  padding: 8px 10px;
  border-radius: 6px;
  border-bottom: none;
}

:deep(.fact-idx) {
  min-width: 22px;
  height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #F3F4F6;
  border-radius: 6px;
  font-size: 10px;
  font-weight: 700;
  color: #6B7280;
  flex-shrink: 0;
}

:deep(.fact-row.active .fact-idx) {
  background: #A7F3D0;
  color: #065F46;
}

:deep(.fact-text) {
  font-size: 12px;
  color: #4B5563;
  line-height: 1.6;
}

/* Entities Section */
:deep(.entities-section) {
  margin-bottom: 14px;
}

:deep(.entity-chips) {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

:deep(.entity-chip) {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  padding: 6px 12px;
}

:deep(.chip-name) {
  font-size: 12px;
  font-weight: 500;
  color: #111827;
}

:deep(.chip-type) {
  font-size: 10px;
  color: #9CA3AF;
  background: #E5E7EB;
  padding: 1px 6px;
  border-radius: 3px;
}

/* Relations Section */
:deep(.relations-section) {
  margin-bottom: 14px;
}

:deep(.relation-row) {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 0;
  flex-wrap: wrap;
  border-bottom: 1px solid #F3F4F6;
}

:deep(.relation-row:last-child) {
  border-bottom: none;
}

:deep(.rel-node) {
  font-size: 12px;
  font-weight: 500;
  color: #111827;
  background: #F3F4F6;
  padding: 4px 10px;
  border-radius: 4px;
}

:deep(.rel-edge) {
  font-size: 10px;
  font-weight: 600;
  color: #FFFFFF;
  background: #4F46E5;
  padding: 3px 10px;
  border-radius: 10px;
}

/* ========== Interview Display - Conversation Style ========== */
:deep(.interview-display) {
  padding: 0;
}

/* Header */
:deep(.interview-display .interview-header) {
  padding: 0;
  background: transparent;
  border-bottom: none;
  margin-bottom: 16px;
}

:deep(.interview-display .header-main) {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

:deep(.interview-display .header-title) {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 600;
  color: #111827;
  letter-spacing: -0.01em;
}

:deep(.interview-display .header-stats) {
  display: flex;
  align-items: center;
  gap: 6px;
}

:deep(.interview-display .stat-item) {
  display: flex;
  align-items: baseline;
  gap: 4px;
}

:deep(.interview-display .stat-value) {
  font-size: 14px;
  font-weight: 600;
  color: #4F46E5;
  font-family: 'JetBrains Mono', monospace;
}

:deep(.interview-display .stat-label) {
  font-size: 11px;
  color: #9CA3AF;
  text-transform: lowercase;
}

:deep(.interview-display .stat-divider) {
  color: #D1D5DB;
  font-size: 12px;
}

:deep(.interview-display .stat-size) {
  font-size: 11px;
  color: #9CA3AF;
  font-family: 'JetBrains Mono', monospace;
}

:deep(.interview-display .header-topic) {
  margin-top: 4px;
  font-size: 12px;
  color: #6B7280;
  line-height: 1.5;
}

/* Agent Tabs - Card Style */
:deep(.interview-display .agent-tabs) {
  display: flex;
  gap: 8px;
  padding: 0 0 14px 0;
  background: transparent;
  border-bottom: 1px solid #F3F4F6;
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-width: thin;
  scrollbar-color: #E5E7EB transparent;
}

:deep(.interview-display .agent-tabs::-webkit-scrollbar) {
  height: 4px;
}

:deep(.interview-display .agent-tabs::-webkit-scrollbar-track) {
  background: transparent;
}

:deep(.interview-display .agent-tabs::-webkit-scrollbar-thumb) {
  background: #E5E7EB;
  border-radius: 2px;
}

:deep(.interview-display .agent-tabs::-webkit-scrollbar-thumb:hover) {
  background: #D1D5DB;
}

:deep(.interview-display .agent-tab) {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 500;
  color: #6B7280;
  cursor: pointer;
  transition: all 0.15s ease;
  white-space: nowrap;
}

:deep(.interview-display .agent-tab:hover) {
  background: #F3F4F6;
  border-color: #D1D5DB;
  color: #374151;
}

:deep(.interview-display .agent-tab.active) {
  background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
  border-color: #A5B4FC;
  color: #4338CA;
  box-shadow: 0 1px 2px rgba(99, 102, 241, 0.1);
}

:deep(.interview-display .tab-avatar) {
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #E5E7EB;
  color: #6B7280;
  font-size: 10px;
  font-weight: 700;
  border-radius: 50%;
  flex-shrink: 0;
}

:deep(.interview-display .agent-tab:hover .tab-avatar) {
  background: #D1D5DB;
}

:deep(.interview-display .agent-tab.active .tab-avatar) {
  background: #6366F1;
  color: #FFFFFF;
}

:deep(.interview-display .tab-name) {
  max-width: 100px;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Interview Detail */
:deep(.interview-display .interview-detail) {
  padding: 12px 0;
  background: transparent;
}

/* Agent Profile - No card */
:deep(.interview-display .agent-profile) {
  display: flex;
  gap: 12px;
  padding: 0;
  background: transparent;
  border: none;
  margin-bottom: 16px;
}

:deep(.interview-display .profile-avatar) {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #E5E7EB;
  color: #6B7280;
  font-size: 14px;
  font-weight: 600;
  border-radius: 50%;
  flex-shrink: 0;
}

:deep(.interview-display .profile-info) {
  flex: 1;
  min-width: 0;
}

:deep(.interview-display .profile-name) {
  font-size: 13px;
  font-weight: 600;
  color: #111827;
  margin-bottom: 2px;
}

:deep(.interview-display .profile-role) {
  font-size: 11px;
  color: #6B7280;
  margin-bottom: 4px;
}

:deep(.interview-display .profile-bio) {
  font-size: 11px;
  color: #9CA3AF;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Selection Reason */
:deep(.interview-display .selection-reason) {
  background: #F8FAFC;
  border: 1px solid #E2E8F0;
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 16px;
}

:deep(.interview-display .reason-label) {
  font-size: 11px;
  font-weight: 600;
  color: #64748B;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: 6px;
}

:deep(.interview-display .reason-content) {
  font-size: 12px;
  color: #475569;
  line-height: 1.6;
}

/* Q&A Thread - Clean list */
:deep(.interview-display .qa-thread) {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

:deep(.interview-display .qa-pair) {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 0;
}

:deep(.interview-display .qa-question),
:deep(.interview-display .qa-answer) {
  display: flex;
  gap: 12px;
}

:deep(.interview-display .qa-badge) {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  border-radius: 4px;
  flex-shrink: 0;
}

:deep(.interview-display .q-badge) {
  background: transparent;
  color: #9CA3AF;
  border: 1px solid #E5E7EB;
}

:deep(.interview-display .a-badge) {
  background: #4F46E5;
  color: #FFFFFF;
  border: 1px solid #4F46E5;
}

:deep(.interview-display .qa-content) {
  flex: 1;
  min-width: 0;
}

:deep(.interview-display .qa-sender) {
  font-size: 11px;
  font-weight: 600;
  color: #9CA3AF;
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

:deep(.interview-display .qa-text) {
  font-size: 13px;
  color: #374151;
  line-height: 1.6;
}

:deep(.interview-display .qa-answer) {
  background: transparent;
  padding: 0;
  border: none;
  margin-top: 0;
}

:deep(.interview-display .answer-placeholder) {
  opacity: 0.6;
}

:deep(.interview-display .placeholder-text) {
  font-style: italic;
  color: #9CA3AF;
}

:deep(.interview-display .qa-answer-header) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}

/* Platform Switch */
:deep(.interview-display .platform-switch) {
  display: flex;
  gap: 2px;
  background: transparent;
  padding: 0;
  border-radius: 0;
}

:deep(.interview-display .platform-btn) {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 500;
  color: #9CA3AF;
  cursor: pointer;
  transition: all 0.15s ease;
}

:deep(.interview-display .platform-btn:hover) {
  color: #6B7280;
}

:deep(.interview-display .platform-btn.active) {
  background: transparent;
  color: #4F46E5;
  border-color: #E5E7EB;
  box-shadow: none;
}

:deep(.interview-display .platform-icon) {
  flex-shrink: 0;
}

:deep(.interview-display .answer-text) {
  font-size: 13px;
  color: #111827;
  line-height: 1.6;
}

:deep(.interview-display .answer-text strong) {
  color: #111827;
  font-weight: 600;
}

:deep(.interview-display .expand-answer-btn) {
  display: inline-block;
  margin-top: 8px;
  padding: 0;
  background: transparent;
  border: none;
  border-bottom: 1px dotted #D1D5DB;
  border-radius: 0;
  font-size: 11px;
  font-weight: 500;
  color: #9CA3AF;
  cursor: pointer;
  transition: all 0.15s ease;
}

:deep(.interview-display .expand-answer-btn:hover) {
  background: transparent;
  color: #6B7280;
  border-bottom-style: solid;
}

/* Quotes Section - Clean list */
:deep(.interview-display .quotes-section) {
  background: transparent;
  border: none;
  border-top: 1px solid #F3F4F6;
  border-radius: 0;
  padding: 16px 0 0 0;
  margin-top: 16px;
}

:deep(.interview-display .quotes-header) {
  font-size: 11px;
  font-weight: 600;
  color: #9CA3AF;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 12px;
}

:deep(.interview-display .quotes-list) {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

:deep(.interview-display .quote-item) {
  margin: 0;
  padding: 10px 12px;
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  font-size: 12px;
  font-style: italic;
  color: #4B5563;
  line-height: 1.5;
}

/* Summary Section */
:deep(.interview-display .summary-section) {
  margin-top: 20px;
  padding: 16px 0 0 0;
  background: transparent;
  border: none;
  border-top: 1px solid #F3F4F6;
  border-radius: 0;
}

:deep(.interview-display .summary-header) {
  font-size: 11px;
  font-weight: 600;
  color: #9CA3AF;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 8px;
}

:deep(.interview-display .summary-content) {
  font-size: 13px;
  color: #374151;
  line-height: 1.6;
}

/* Markdown styles in summary */
:deep(.interview-display .summary-content h2),
:deep(.interview-display .summary-content h3),
:deep(.interview-display .summary-content h4),
:deep(.interview-display .summary-content h5) {
  margin: 12px 0 8px 0;
  font-weight: 600;
  color: #111827;
}

:deep(.interview-display .summary-content h2) {
  font-size: 15px;
}

:deep(.interview-display .summary-content h3) {
  font-size: 14px;
}

:deep(.interview-display .summary-content h4),
:deep(.interview-display .summary-content h5) {
  font-size: 13px;
}

:deep(.interview-display .summary-content p) {
  margin: 8px 0;
}

:deep(.interview-display .summary-content strong) {
  font-weight: 600;
  color: #111827;
}

:deep(.interview-display .summary-content em) {
  font-style: italic;
}

:deep(.interview-display .summary-content ul),
:deep(.interview-display .summary-content ol) {
  margin: 8px 0;
  padding-left: 20px;
}

:deep(.interview-display .summary-content li) {
  margin: 4px 0;
}

:deep(.interview-display .summary-content blockquote) {
  margin: 8px 0;
  padding-left: 12px;
  border-left: 3px solid #E5E7EB;
  color: #6B7280;
  font-style: italic;
}

/* Markdown styles in quotes */
:deep(.interview-display .quote-item strong) {
  font-weight: 600;
  color: #374151;
}

:deep(.interview-display .quote-item em) {
  font-style: italic;
}

/* ========== Enhanced Insight Display Styles ========== */
:deep(.insight-display) {
  padding: 0;
}

:deep(.insight-header) {
  padding: 12px 16px;
  background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
  border-radius: 8px 8px 0 0;
  border: 1px solid #C4B5FD;
  border-bottom: none;
}

:deep(.insight-header .header-main) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

:deep(.insight-header .header-title) {
  font-size: 14px;
  font-weight: 700;
  color: #6D28D9;
}

:deep(.insight-header .header-stats) {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
}

:deep(.insight-header .stat-item) {
  display: flex;
  align-items: baseline;
  gap: 2px;
}

:deep(.insight-header .stat-value) {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #7C3AED;
}

:deep(.insight-header .stat-label) {
  color: #8B5CF6;
  font-size: 10px;
}

:deep(.insight-header .stat-divider) {
  color: #C4B5FD;
  margin: 0 4px;
}

:deep(.insight-header .stat-size) {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #9CA3AF;
}

:deep(.insight-header .header-topic) {
  font-size: 13px;
  color: #5B21B6;
  line-height: 1.5;
}

:deep(.insight-header .header-scenario) {
  margin-top: 6px;
  font-size: 11px;
  color: #7C3AED;
}

:deep(.insight-header .scenario-label) {
  font-weight: 600;
}

:deep(.insight-tabs) {
  display: flex;
  gap: 2px;
  padding: 8px 12px;
  background: #FAFAFA;
  border: 1px solid #E5E7EB;
  border-top: none;
}

:deep(.insight-tab) {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 500;
  color: #6B7280;
  cursor: pointer;
  transition: all 0.15s ease;
}

:deep(.insight-tab:hover) {
  background: #F3F4F6;
  color: #374151;
}

:deep(.insight-tab.active) {
  background: #FFFFFF;
  color: #7C3AED;
  border-color: #C4B5FD;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}


:deep(.insight-content) {
  padding: 12px;
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-top: none;
  border-radius: 0 0 8px 8px;
}

:deep(.insight-display .panel-header) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid #F3F4F6;
}

:deep(.insight-display .panel-title) {
  font-size: 12px;
  font-weight: 600;
  color: #374151;
}

:deep(.insight-display .panel-count) {
  font-size: 10px;
  color: #9CA3AF;
}

:deep(.insight-display .facts-list),
:deep(.insight-display .relations-list),
:deep(.insight-display .subqueries-list) {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

:deep(.insight-display .entities-grid) {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

:deep(.insight-display .fact-item) {
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
}

:deep(.insight-display .fact-number) {
  flex-shrink: 0;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #E5E7EB;
  border-radius: 50%;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  color: #6B7280;
}

:deep(.insight-display .fact-content) {
  flex: 1;
  font-size: 12px;
  color: #374151;
  line-height: 1.6;
}

/* Entity Tag Styles - Compact multi-column layout */
:deep(.insight-display .entity-tag) {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  cursor: default;
  transition: all 0.15s ease;
}

:deep(.insight-display .entity-tag:hover) {
  background: #F3F4F6;
  border-color: #D1D5DB;
}

:deep(.insight-display .entity-tag .entity-name) {
  font-size: 12px;
  font-weight: 500;
  color: #111827;
}

:deep(.insight-display .entity-tag .entity-type) {
  font-size: 9px;
  color: #7C3AED;
  background: #EDE9FE;
  padding: 1px 4px;
  border-radius: 3px;
}

:deep(.insight-display .entity-tag .entity-fact-count) {
  font-size: 9px;
  color: #9CA3AF;
  margin-left: 2px;
}

/* Legacy entity card styles for backwards compatibility */
:deep(.insight-display .entity-card) {
  padding: 12px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
}

:deep(.insight-display .entity-header) {
  display: flex;
  align-items: center;
  gap: 10px;
}

:deep(.insight-display .entity-info) {
  flex: 1;
}

:deep(.insight-display .entity-card .entity-name) {
  font-size: 13px;
  font-weight: 600;
  color: #111827;
}

:deep(.insight-display .entity-card .entity-type) {
  font-size: 10px;
  color: #7C3AED;
  background: #EDE9FE;
  padding: 2px 6px;
  border-radius: 4px;
  display: inline-block;
  margin-top: 2px;
}

:deep(.insight-display .entity-card .entity-fact-count) {
  font-size: 10px;
  color: #9CA3AF;
  background: #F3F4F6;
  padding: 2px 6px;
  border-radius: 4px;
}

:deep(.insight-display .entity-summary) {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #E5E7EB;
  font-size: 11px;
  color: #6B7280;
  line-height: 1.5;
}

/* Relation Item Styles */
:deep(.insight-display .relation-item) {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
}

:deep(.insight-display .rel-source),
:deep(.insight-display .rel-target) {
  padding: 4px 8px;
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  color: #374151;
}

:deep(.insight-display .rel-arrow) {
  display: flex;
  align-items: center;
  gap: 4px;
  flex: 1;
}

:deep(.insight-display .rel-line) {
  flex: 1;
  height: 1px;
  background: #D1D5DB;
}

:deep(.insight-display .rel-label) {
  padding: 2px 6px;
  background: #EDE9FE;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 500;
  color: #7C3AED;
  white-space: nowrap;
}

/* Sub-query Styles */
:deep(.insight-display .subquery-item) {
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
}

:deep(.insight-display .subquery-number) {
  flex-shrink: 0;
  padding: 2px 6px;
  background: #7C3AED;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  color: #FFFFFF;
}

:deep(.insight-display .subquery-text) {
  font-size: 12px;
  color: #374151;
  line-height: 1.5;
}

/* Expand Button */
:deep(.insight-display .expand-btn),
:deep(.panorama-display .expand-btn),
:deep(.quick-search-display .expand-btn) {
  display: block;
  width: 100%;
  margin-top: 12px;
  padding: 8px 12px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 500;
  color: #6B7280;
  cursor: pointer;
  transition: all 0.15s ease;
  text-align: center;
}

:deep(.insight-display .expand-btn:hover),
:deep(.panorama-display .expand-btn:hover),
:deep(.quick-search-display .expand-btn:hover) {
  background: #F3F4F6;
  color: #374151;
  border-color: #D1D5DB;
}

/* Empty State */
:deep(.insight-display .empty-state),
:deep(.panorama-display .empty-state),
:deep(.quick-search-display .empty-state) {
  padding: 24px;
  text-align: center;
  font-size: 12px;
  color: #9CA3AF;
}

/* ========== Enhanced Panorama Display Styles ========== */
:deep(.panorama-display) {
  padding: 0;
}

:deep(.panorama-header) {
  padding: 12px 16px;
  background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
  border-radius: 8px 8px 0 0;
  border: 1px solid #93C5FD;
  border-bottom: none;
}

:deep(.panorama-header .header-main) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

:deep(.panorama-header .header-title) {
  font-size: 14px;
  font-weight: 700;
  color: #1D4ED8;
}

:deep(.panorama-header .header-stats) {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
}

:deep(.panorama-header .stat-item) {
  display: flex;
  align-items: baseline;
  gap: 2px;
}

:deep(.panorama-header .stat-value) {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #2563EB;
}

:deep(.panorama-header .stat-label) {
  color: #60A5FA;
  font-size: 10px;
}

:deep(.panorama-header .stat-divider) {
  color: #93C5FD;
  margin: 0 4px;
}

:deep(.panorama-header .stat-size) {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #9CA3AF;
}

:deep(.panorama-header .header-topic) {
  font-size: 13px;
  color: #1E40AF;
  line-height: 1.5;
}

:deep(.panorama-tabs) {
  display: flex;
  gap: 2px;
  padding: 8px 12px;
  background: #FAFAFA;
  border: 1px solid #E5E7EB;
  border-top: none;
}

:deep(.panorama-tab) {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 500;
  color: #6B7280;
  cursor: pointer;
  transition: all 0.15s ease;
}

:deep(.panorama-tab:hover) {
  background: #F3F4F6;
  color: #374151;
}

:deep(.panorama-tab.active) {
  background: #FFFFFF;
  color: #2563EB;
  border-color: #93C5FD;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}


:deep(.panorama-content) {
  padding: 12px;
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-top: none;
  border-radius: 0 0 8px 8px;
}

:deep(.panorama-display .panel-header) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid #F3F4F6;
}

:deep(.panorama-display .panel-title) {
  font-size: 12px;
  font-weight: 600;
  color: #374151;
}

:deep(.panorama-display .panel-count) {
  font-size: 10px;
  color: #9CA3AF;
}

:deep(.panorama-display .facts-list) {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

:deep(.panorama-display .fact-item) {
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
}

:deep(.panorama-display .fact-item.active) {
  background: #F9FAFB;
  border-color: #E5E7EB;
}

:deep(.panorama-display .fact-item.historical) {
  background: #F9FAFB;
  border-color: #E5E7EB;
}

:deep(.panorama-display .fact-number) {
  flex-shrink: 0;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #E5E7EB;
  border-radius: 50%;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  color: #6B7280;
}

:deep(.panorama-display .fact-item.active .fact-number) {
  background: #E5E7EB;
  color: #6B7280;
}

:deep(.panorama-display .fact-item.historical .fact-number) {
  background: #9CA3AF;
  color: #FFFFFF;
}

:deep(.panorama-display .fact-content) {
  flex: 1;
  font-size: 12px;
  color: #374151;
  line-height: 1.6;
}

:deep(.panorama-display .fact-time) {
  display: block;
  font-size: 10px;
  color: #9CA3AF;
  margin-bottom: 4px;
  font-family: 'JetBrains Mono', monospace;
}

:deep(.panorama-display .fact-text) {
  display: block;
}

/* Entities Grid */
:deep(.panorama-display .entities-grid) {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

:deep(.panorama-display .entity-tag) {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
}

:deep(.panorama-display .entity-name) {
  font-size: 12px;
  font-weight: 500;
  color: #374151;
}

:deep(.panorama-display .entity-type) {
  font-size: 10px;
  color: #2563EB;
  background: #DBEAFE;
  padding: 2px 6px;
  border-radius: 4px;
}

/* ========== Enhanced Quick Search Display Styles ========== */
:deep(.quick-search-display) {
  padding: 0;
}

:deep(.quicksearch-header) {
  padding: 12px 16px;
  background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
  border-radius: 8px 8px 0 0;
  border: 1px solid #FDBA74;
  border-bottom: none;
}

:deep(.quicksearch-header .header-main) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

:deep(.quicksearch-header .header-title) {
  font-size: 14px;
  font-weight: 700;
  color: #C2410C;
}

:deep(.quicksearch-header .header-stats) {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
}

:deep(.quicksearch-header .stat-item) {
  display: flex;
  align-items: baseline;
  gap: 2px;
}

:deep(.quicksearch-header .stat-value) {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #EA580C;
}

:deep(.quicksearch-header .stat-label) {
  color: #FB923C;
  font-size: 10px;
}

:deep(.quicksearch-header .stat-divider) {
  color: #FDBA74;
  margin: 0 4px;
}

:deep(.quicksearch-header .stat-size) {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #9CA3AF;
}

:deep(.quicksearch-header .header-query) {
  font-size: 13px;
  color: #9A3412;
  line-height: 1.5;
}

:deep(.quicksearch-header .query-label) {
  font-weight: 600;
}

:deep(.quicksearch-tabs) {
  display: flex;
  gap: 2px;
  padding: 8px 12px;
  background: #FAFAFA;
  border: 1px solid #E5E7EB;
  border-top: none;
}

:deep(.quicksearch-tab) {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 500;
  color: #6B7280;
  cursor: pointer;
  transition: all 0.15s ease;
}

:deep(.quicksearch-tab:hover) {
  background: #F3F4F6;
  color: #374151;
}

:deep(.quicksearch-tab.active) {
  background: #FFFFFF;
  color: #EA580C;
  border-color: #FDBA74;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}


:deep(.quicksearch-content) {
  padding: 12px;
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-top: none;
  border-radius: 0 0 8px 8px;
}

/* When there are no tabs, content connects directly to header */
:deep(.quicksearch-content.no-tabs) {
  border-top: none;
}

:deep(.quick-search-display .panel-header) {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid #F3F4F6;
}

:deep(.quick-search-display .panel-title) {
  font-size: 12px;
  font-weight: 600;
  color: #374151;
}

:deep(.quick-search-display .panel-count) {
  font-size: 10px;
  color: #9CA3AF;
}

:deep(.quick-search-display .facts-list) {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

:deep(.quick-search-display .fact-item) {
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
}

:deep(.quick-search-display .fact-item.active) {
  background: #F9FAFB;
  border-color: #E5E7EB;
}

:deep(.quick-search-display .fact-number) {
  flex-shrink: 0;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #E5E7EB;
  border-radius: 50%;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  color: #6B7280;
}

:deep(.quick-search-display .fact-item.active .fact-number) {
  background: #E5E7EB;
  color: #6B7280;
}

:deep(.quick-search-display .fact-content) {
  flex: 1;
  font-size: 12px;
  color: #374151;
  line-height: 1.6;
}

/* Edges Panel */
:deep(.quick-search-display .edges-list) {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

:deep(.quick-search-display .edge-item) {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
}

:deep(.quick-search-display .edge-source),
:deep(.quick-search-display .edge-target) {
  padding: 4px 8px;
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  color: #374151;
}

:deep(.quick-search-display .edge-arrow) {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
}

:deep(.quick-search-display .edge-line) {
  flex: 1;
  height: 1px;
  background: #D1D5DB;
}

:deep(.quick-search-display .edge-label) {
  padding: 2px 6px;
  background: #FFEDD5;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 500;
  color: #C2410C;
  white-space: nowrap;
}

/* Nodes Grid */
:deep(.quick-search-display .nodes-grid) {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

:deep(.quick-search-display .node-tag) {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
}

:deep(.quick-search-display .node-name) {
  font-size: 12px;
  font-weight: 500;
  color: #374151;
}

:deep(.quick-search-display .node-type) {
  font-size: 10px;
  color: #EA580C;
  background: #FFEDD5;
  padding: 2px 6px;
  border-radius: 4px;
}

/* Console Logs - Keep consistent with Step3Simulation.vue */
.console-logs {
  background: #000;
  color: #DDD;
  padding: 16px;
  font-family: 'JetBrains Mono', monospace;
  border-top: 1px solid #222;
  flex-shrink: 0;
  transition: padding 0.3s ease;
}

.log-header {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid #333;
  padding-bottom: 8px;
  margin-bottom: 8px;
  font-size: 10px;
  color: #666;
  cursor: pointer;
  user-select: none;
  transition: color 0.2s ease, border-color 0.3s ease, margin-bottom 0.3s ease, padding-bottom 0.3s ease;
}

.log-header:hover {
  color: #999;
}

.log-title-container {
  display: flex;
  align-items: center;
  gap: 6px;
}

.console-collapse-icon {
  color: #666;
  transition: transform 0.3s ease, color 0.2s ease;
  flex-shrink: 0;
}

.console-collapse-icon.is-collapsed {
  transform: rotate(-90deg);
}

.log-header:hover .console-collapse-icon {
  color: #999;
}

.console-logs.is-collapsed {
  padding: 10px 16px;
}

.console-logs.is-collapsed .log-header {
  border-bottom: 1px solid transparent;
  margin-bottom: 0;
  padding-bottom: 0;
}

.log-title {
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.log-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
  height: 100px;
  overflow-y: auto;
  padding-right: 4px;
}

.log-content::-webkit-scrollbar { width: 4px; }
.log-content::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

.log-line {
  font-size: 11px;
  line-height: 1.5;
}

.log-msg {
  color: #BBB;
  word-break: break-all;
}

.log-msg.error { color: #EF5350; }
.log-msg.warning { color: #FFA726; }
.log-msg.success { color: #66BB6A; }

/* Regenerate Modal Styles */
.regenerate-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.4);
  backdrop-filter: blur(2px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10000;
}

.regenerate-modal {
  background: #FFF;
  border-radius: 12px;
  width: 100%;
  max-width: 500px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  padding: 24px;
}

.modal-enter-active,
.modal-leave-active {
  transition: opacity 0.2s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}

.modal-enter-active .regenerate-modal {
  animation: modal-pop 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.modal-leave-active .regenerate-modal {
  transition: transform 0.2s ease;
  transform: scale(0.95);
}

@keyframes modal-pop {
  0% { transform: scale(0.95); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
</style>

<style>
/* English locale: smaller report title */
html[lang="en"] .report-header-block .main-title {
  font-size: 28px;
}
</style>
