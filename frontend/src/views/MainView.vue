<template>
  <div class="main-view">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <div class="brand" @click="router.push('/')">MIROFISH</div>
      </div>
      
      <div class="header-center">
        <div class="view-switcher">
          <button 
            v-for="mode in ['graph', 'split', 'workbench']" 
            :key="mode"
            class="switch-btn"
            :class="{ active: viewMode === mode }"
            @click="viewMode = mode"
          >
            {{ { graph: $t('main.layoutGraph'), split: $t('main.layoutSplit'), workbench: $t('main.layoutWorkbench') }[mode] }}
          </button>
        </div>
      </div>

      <div class="header-right">
        <LanguageSwitcher />
        <div class="step-divider"></div>
        <div class="workflow-step">
          <span class="step-num">Step {{ currentStep }}/5</span>
          <span class="step-name">{{ $tm('main.stepNames')[currentStep - 1] }}</span>
        </div>
        <div class="step-divider"></div>
        <span class="status-indicator" :class="statusClass">
          <span class="dot"></span>
          {{ statusText }}
        </span>
      </div>
    </header>

    <!-- Main Content Area -->
    <main class="content-area">
      <!-- Left Panel: Graph -->
      <div class="panel-wrapper left" :style="leftPanelStyle">
        <GraphPanel 
          :graphData="graphData"
          :loading="graphLoading"
          :currentPhase="currentPhase"
          @refresh="refreshGraph"
          @toggle-maximize="toggleMaximize('graph')"
        />
      </div>

      <!-- Right Panel: Step Components -->
      <div class="panel-wrapper right" :style="rightPanelStyle">
        <!-- Step 1: 图谱构建 -->
        <Step1GraphBuild 
          v-if="currentStep === 1"
          :currentPhase="currentPhase"
          :projectData="projectData"
          :ontologyProgress="ontologyProgress"
          :buildProgress="buildProgress"
          :graphData="graphData"
          :systemLogs="systemLogs"
          :stepErrors="stepErrors"
          @next-step="handleNextStep"
          @retry-ontology="retryOntology"
          @retry-build="retryBuildGraph"
        />
        <!-- Step 2: 环境搭建 -->
        <Step2EnvSetup
          v-else-if="currentStep === 2"
          :projectData="projectData"
          :graphData="graphData"
          :systemLogs="systemLogs"
          @go-back="handleGoBack"
          @next-step="handleNextStep"
          @add-log="addLog"
        />
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import GraphPanel from '../components/GraphPanel.vue'
import Step1GraphBuild from '../components/Step1GraphBuild.vue'
import Step2EnvSetup from '../components/Step2EnvSetup.vue'
import { generateOntology, getProject, buildGraph, getTaskStatus, getGraphData } from '../api/graph'
import { createSimulation } from '../api/simulation'
import { getPendingUpload, clearPendingUpload } from '../store/pendingUpload'
import { usePipelineAutopilot } from '../composables/usePipelineAutopilot'
import LanguageSwitcher from '../components/LanguageSwitcher.vue'

const route = useRoute()
const router = useRouter()
const { t, tm } = useI18n()
const { options: runOptions, shouldAdvance, markAdvanced } = usePipelineAutopilot()

// Layout State
const viewMode = ref('split') // graph | split | workbench

// Step State
const currentStep = ref(1) // 1: 图谱构建, 2: 环境搭建, 3: 开始模拟, 4: 报告生成, 5: 深度互动
const stepNames = computed(() => tm('main.stepNames'))

// Data State
const currentProjectId = ref(route.params.projectId)
const loading = ref(false)
const graphLoading = ref(false)
const error = ref('')
const projectData = ref(null)
const graphData = ref(null)
const currentPhase = ref(-1) // -1: Upload, 0: Ontology, 1: Build, 2: Complete
const ontologyProgress = ref(null)
const buildProgress = ref(null)
const systemLogs = ref([])
const stepErrors = ref({ ontology: '', build: '' })

// Polling timers
let pollTimer = null
let graphPollTimer = null
let ontologyPollTimer = null
let lastOntologyLogMessage = ''
let lastBuildLogMessage = ''

const applyBuildProgress = (progress, message) => {
  const prev = buildProgress.value?.progress ?? 0
  const next = Math.max(prev, Number(progress) || 0)
  buildProgress.value = {
    progress: next,
    message: message || buildProgress.value?.message || 'Building graph...',
  }
}

const refreshBuildProgressFromProject = async () => {
  if (!currentProjectId.value || currentProjectId.value === 'new') return
  try {
    const res = await getProject(currentProjectId.value)
    if (!res.success) return
    const p = res.data
    if (p.status === 'graph_building' && (p.graph_build_progress || p.graph_build_message)) {
      applyBuildProgress(p.graph_build_progress, p.graph_build_message)
    }
    if (p.graph_build_task_stale) {
      addLog(
        'Build task is no longer on the server (often after a restart). Use Retry build to continue.',
        'error',
      )
    }
  } catch (e) {
    console.warn('refreshBuildProgressFromProject:', e)
  }
}

// --- Computed Layout Styles ---
const leftPanelStyle = computed(() => {
  if (viewMode.value === 'graph') return { width: '100%', opacity: 1, transform: 'translateX(0)' }
  if (viewMode.value === 'workbench') return { width: '0%', opacity: 0, transform: 'translateX(-20px)' }
  return { width: '50%', opacity: 1, transform: 'translateX(0)' }
})

const rightPanelStyle = computed(() => {
  if (viewMode.value === 'workbench') return { width: '100%', opacity: 1, transform: 'translateX(0)' }
  if (viewMode.value === 'graph') return { width: '0%', opacity: 0, transform: 'translateX(20px)' }
  return { width: '50%', opacity: 1, transform: 'translateX(0)' }
})

// --- Status Computed ---
const statusClass = computed(() => {
  if (error.value) return 'error'
  if (currentPhase.value >= 2) return 'completed'
  return 'processing'
})

const statusText = computed(() => {
  if (error.value) return 'Error'
  if (currentPhase.value >= 2) return 'Ready'
  if (currentPhase.value === 1) return 'Building Graph'
  if (currentPhase.value === 0) return 'Generating Ontology'
  return 'Initializing'
})

// --- Helpers ---
const addLog = (msg, level = 'info') => {
  const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + '.' + new Date().getMilliseconds().toString().padStart(3, '0')
  systemLogs.value.push({ time, msg, level })
  if (systemLogs.value.length > 100) {
    systemLogs.value.shift()
  }
}

// --- Layout Methods ---
const toggleMaximize = (target) => {
  if (viewMode.value === target) {
    viewMode.value = 'split'
  } else {
    viewMode.value = target
  }
}

const handleNextStep = (params = {}) => {
  if (currentStep.value < 5) {
    currentStep.value++
    addLog(t('log.enterStep', { step: currentStep.value, name: stepNames.value[currentStep.value - 1] }))
    
    // 如果是从 Step 2 进入 Step 3，记录模拟轮数配置
    if (currentStep.value === 3 && params.maxRounds) {
      addLog(t('log.customSimRounds', { rounds: params.maxRounds }))
    }
  }
}

const handleGoBack = () => {
  if (currentStep.value > 1) {
    currentStep.value--
    addLog(t('log.returnToStep', { step: currentStep.value, name: stepNames.value[currentStep.value - 1] }))
  }
}

const retryOntology = async () => {
  if (!currentProjectId.value || currentProjectId.value === 'new') return
  stopOntologyPolling()
  stepErrors.value.ontology = ''
  error.value = ''
  currentPhase.value = 0
  ontologyProgress.value = { message: 'Retrying ontology generation...', progress: 0 }
  addLog('Retrying ontology generation...')
  lastOntologyLogMessage = ''

  const formData = new FormData()
  formData.append('simulation_requirement', projectData.value?.simulation_requirement || '')
  formData.append('project_id', currentProjectId.value)

  try {
    const res = await generateOntology(formData)
    if (res.success && res.data?.task_id) {
      addLog(`Ontology retry task started: ${res.data.task_id}`)
      startOntologyPolling(res.data.task_id)
    } else if (res.success && res.data?.ontology) {
      projectData.value = res.data
      ontologyProgress.value = null
      addLog('Ontology retry completed.')
      await startBuildGraph()
    } else {
      const errMsg = res.error || 'Ontology retry failed'
      stepErrors.value.ontology = errMsg
      error.value = errMsg
      ontologyProgress.value = null
      addLog(`Ontology retry failed: ${errMsg}`, 'error')
    }
  } catch (err) {
    const errMsg = err.message || 'Ontology retry error'
    stepErrors.value.ontology = errMsg
    error.value = errMsg
    ontologyProgress.value = null
    addLog(`Ontology retry exception: ${errMsg}`, 'error')
  }
}

const retryBuildGraph = async () => {
  stopPolling()
  stopGraphPolling()
  stepErrors.value.build = ''
  error.value = ''
  buildProgress.value = null
  await startBuildGraph()
}

// --- Data Logic ---

const initProject = async () => {
  addLog('Project view initialized.')
  if (currentProjectId.value === 'new') {
    await handleNewProject()
  } else {
    await loadProject()
  }
}

const handleNewProject = async () => {
  const pending = getPendingUpload()
  if (!pending.isPending || (pending.files.length === 0 && !pending.useVertexSearch)) {
    error.value = 'No pending seed data (add files or enable Gemini web search).'
    addLog('Error: No pending files or Gemini web search flag for new project.')
    return
  }
  
  try {
    loading.value = true
    currentPhase.value = 0
    ontologyProgress.value = { message: 'Uploading files and starting ontology task...', progress: 0 }
    addLog('Starting ontology generation: uploading files...')
    lastOntologyLogMessage = ''
    
    const formData = new FormData()
    pending.files.forEach(f => formData.append('files', f))
    formData.append('simulation_requirement', pending.simulationRequirement)
    formData.append('use_vertex_search', pending.useVertexSearch ? 'true' : 'false')
    
    const res = await generateOntology(formData)
    if (res.success && res.data?.task_id) {
      clearPendingUpload()
      currentProjectId.value = res.data.project_id
      addLog(`Ontology task started: ${res.data.task_id}`, 'info')
      addLog(`Project created: ${res.data.project_id}`, 'info')
      router.replace({ name: 'Process', params: { projectId: res.data.project_id } })
      startOntologyPolling(res.data.task_id)
    } else if (res.success && res.data?.ontology) {
      // Legacy synchronous response (if server returns full payload)
      clearPendingUpload()
      currentProjectId.value = res.data.project_id
      projectData.value = res.data
      router.replace({ name: 'Process', params: { projectId: res.data.project_id } })
      ontologyProgress.value = null
      addLog(`Ontology generated for project ${res.data.project_id}`)
      await startBuildGraph()
    } else {
      error.value = res.error || 'Ontology generation failed'
      ontologyProgress.value = null
      addLog(`Error starting ontology: ${error.value}`, 'error')
    }
  } catch (err) {
    error.value = err.message
    ontologyProgress.value = null
    addLog(`Exception in handleNewProject: ${err.message}`, 'error')
  } finally {
    loading.value = false
  }
}

const loadProject = async () => {
  try {
    loading.value = true
    addLog(`Loading project ${currentProjectId.value}...`)
    const res = await getProject(currentProjectId.value)
    if (res.success) {
      projectData.value = res.data
      updatePhaseByStatus(res.data.status)
      addLog(`Project loaded. Status: ${res.data.status}`)

      if (res.data.ontology_task_id) {
        currentPhase.value = 0
        ontologyProgress.value = { message: 'Resuming ontology generation...', progress: 0 }
        addLog(`Resuming ontology task: ${res.data.ontology_task_id}`)
        startOntologyPolling(res.data.ontology_task_id)
      } else if (res.data.status === 'ontology_generated' && !res.data.graph_id) {
        await startBuildGraph()
      } else if (res.data.status === 'graph_building' && res.data.graph_build_task_id) {
        currentPhase.value = 1
        applyBuildProgress(res.data.graph_build_progress, res.data.graph_build_message)
        if (res.data.graph_id) {
          await loadGraph(res.data.graph_id)
        }
        startPollingTask(res.data.graph_build_task_id)
        startGraphPolling()
      } else if (res.data.status === 'graph_completed' && res.data.graph_id) {
        currentPhase.value = 2
        await loadGraph(res.data.graph_id)
      }
    } else {
      error.value = res.error
      addLog(`Error loading project: ${res.error}`)
    }
  } catch (err) {
    error.value = err.message
    addLog(`Exception in loadProject: ${err.message}`)
  } finally {
    loading.value = false
  }
}

const updatePhaseByStatus = (status) => {
  switch (status) {
    case 'created':
    case 'ontology_generated': currentPhase.value = 0; break;
    case 'graph_building': currentPhase.value = 1; break;
    case 'graph_completed': currentPhase.value = 2; break;
    case 'failed': error.value = 'Project failed'; break;
  }
}

const startBuildGraph = async () => {
  try {
    currentPhase.value = 1
    applyBuildProgress(0, 'Starting build...')
    addLog('Initiating graph build...')
    
    const res = await buildGraph({ project_id: currentProjectId.value })
    if (res.success) {
      addLog(`Graph build task started. Task ID: ${res.data.task_id}`)
      startGraphPolling()
      startPollingTask(res.data.task_id)
    } else {
      error.value = res.error
      addLog(`Error starting build: ${res.error}`)
    }
  } catch (err) {
    error.value = err.message
    addLog(`Exception in startBuildGraph: ${err.message}`)
  }
}

const startGraphPolling = () => {
  addLog('Started polling for graph data...')
  fetchGraphData()
  graphPollTimer = setInterval(fetchGraphData, 10000)
}

const fetchGraphData = async () => {
  try {
    // Refresh project info to check for graph_id
    const projRes = await getProject(currentProjectId.value)
    if (projRes.success && projRes.data.graph_id) {
      const gRes = await getGraphData(projRes.data.graph_id)
      if (gRes.success) {
        graphData.value = gRes.data
        const nodeCount = gRes.data.node_count || gRes.data.nodes?.length || 0
        const edgeCount = gRes.data.edge_count || gRes.data.edges?.length || 0
        addLog(`Graph data refreshed. Nodes: ${nodeCount}, Edges: ${edgeCount}`)
      }
    }
  } catch (err) {
    console.warn('Graph fetch error:', err)
  }
}

const startOntologyPolling = (taskId) => {
  pollOntologyTask(taskId)
  ontologyPollTimer = setInterval(() => pollOntologyTask(taskId), 1500)
}

const pollOntologyTask = async (taskId) => {
  try {
    const res = await getTaskStatus(taskId)
    if (!res.success) return

    const task = res.data
    ontologyProgress.value = {
      message: task.message || 'Processing...',
      progress: task.progress ?? 0,
    }

    if (task.message && task.message !== lastOntologyLogMessage) {
      lastOntologyLogMessage = task.message
      addLog(task.message)
    }

    if (task.status === 'completed' && task.result) {
      stopOntologyPolling()
      projectData.value = task.result
      ontologyProgress.value = null
      stepErrors.value.ontology = ''
      addLog(
        `Ontology complete: ${task.result.ontology?.entity_types?.length ?? 0} entity types`,
        'info',
      )
      await startBuildGraph()
    } else if (task.status === 'failed') {
      stopOntologyPolling()
      ontologyProgress.value = null
      const errText = task.error || task.message || 'Ontology generation failed'
      const errMsg = typeof errText === 'string' ? errText.split('\n')[0] : 'Ontology generation failed'
      error.value = errMsg
      stepErrors.value.ontology = errMsg
      addLog(`Ontology failed: ${errMsg}`, 'error')
      if (task.error && task.error !== errMsg) {
        addLog(task.error, 'error')
      }
    }
  } catch (e) {
    addLog(`Ontology poll error: ${e.message}`, 'error')
  }
}

const startPollingTask = (taskId) => {
  lastBuildLogMessage = ''
  pollTaskStatus(taskId)
  pollTimer = setInterval(() => pollTaskStatus(taskId), 2000)
}

const pollTaskStatus = async (taskId) => {
  try {
    const res = await getTaskStatus(taskId)
    if (!res.success) {
      await refreshBuildProgressFromProject()
      return
    }
    const task = res.data
    
    if (task.message && task.message !== lastBuildLogMessage) {
      lastBuildLogMessage = task.message
      addLog(task.message)
    }
    
    applyBuildProgress(task.progress, task.message)
      
    if (task.status === 'completed') {
      addLog('Graph build task completed.')
      stopPolling()
      stopGraphPolling()
      currentPhase.value = 2
      stepErrors.value.build = ''

      const projRes = await getProject(currentProjectId.value)
      if (projRes.success && projRes.data.graph_id) {
          projectData.value = projRes.data
          await loadGraph(projRes.data.graph_id)
          
          if (shouldAdvance('createSimulation')) {
            addLog('Autopilot: Advancing to Environment Setup...')
            markAdvanced('createSimulation')
            try {
              const res = await createSimulation({
                project_id: projectData.value.project_id,
                graph_id: projectData.value.graph_id,
                enable_twitter: runOptions.enableTwitter,
                enable_reddit: runOptions.enableReddit
              })
              if (res.success && res.data?.simulation_id) {
                router.push({ name: 'Simulation', params: { simulationId: res.data.simulation_id } })
              } else {
                addLog(`Autopilot create simulation failed: ${res.error || 'Unknown error'}`, 'error')
              }
            } catch (err) {
              addLog(`Autopilot create simulation exception: ${err.message}`, 'error')
            }
          }
      }
    } else if (task.status === 'failed') {
      stopPolling()
      const errMsg = task.error || task.message || 'Graph build failed'
      error.value = errMsg
      stepErrors.value.build = errMsg
      addLog(`Graph build task failed: ${errMsg}`, 'error')
    }
  } catch (e) {
    console.error(e)
    await refreshBuildProgressFromProject()
  }
}

const loadGraph = async (graphId) => {
  graphLoading.value = true
  addLog(`Loading full graph data: ${graphId}`)
  try {
    const res = await getGraphData(graphId)
    if (res.success) {
      graphData.value = res.data
      addLog('Graph data loaded successfully.')
    } else {
      addLog(`Failed to load graph data: ${res.error}`)
    }
  } catch (e) {
    addLog(`Exception loading graph: ${e.message}`)
  } finally {
    graphLoading.value = false
  }
}

const refreshGraph = () => {
  if (projectData.value?.graph_id) {
    addLog('Manual graph refresh triggered.')
    loadGraph(projectData.value.graph_id)
  }
}

const stopOntologyPolling = () => {
  if (ontologyPollTimer) {
    clearInterval(ontologyPollTimer)
    ontologyPollTimer = null
  }
}

const stopPolling = () => {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

const stopGraphPolling = () => {
  if (graphPollTimer) {
    clearInterval(graphPollTimer)
    graphPollTimer = null
    addLog('Graph polling stopped.')
  }
}

onMounted(() => {
  initProject()
})

onUnmounted(() => {
  stopOntologyPolling()
  stopPolling()
  stopGraphPolling()
})
</script>

<style scoped>
.main-view {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #FFF;
  overflow: hidden;
  font-family: 'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif;
}

/* Header */
.app-header {
  height: 60px;
  border-bottom: 1px solid #EAEAEA;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: #FFF;
  z-index: 100;
  position: relative;
}

.header-center {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
}

.brand {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 800;
  font-size: 18px;
  letter-spacing: 1px;
  cursor: pointer;
}

.view-switcher {
  display: flex;
  background: #F5F5F5;
  padding: 4px;
  border-radius: 6px;
  gap: 4px;
}

.switch-btn {
  border: none;
  background: transparent;
  padding: 6px 16px;
  font-size: 12px;
  font-weight: 600;
  color: #666;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.switch-btn.active {
  background: #FFF;
  color: #000;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #666;
  font-weight: 500;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.workflow-step {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.step-num {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #999;
}

.step-name {
  font-weight: 700;
  color: #000;
}

.step-divider {
  width: 1px;
  height: 14px;
  background-color: #E0E0E0;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #CCC;
}

.status-indicator.processing .dot { background: #FF5722; animation: pulse 1s infinite; }
.status-indicator.completed .dot { background: #4CAF50; }
.status-indicator.error .dot { background: #F44336; }

@keyframes pulse { 50% { opacity: 0.5; } }

/* Content */
.content-area {
  flex: 1;
  display: flex;
  position: relative;
  overflow: hidden;
}

.panel-wrapper {
  height: 100%;
  overflow: hidden;
  transition: width 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), opacity 0.3s ease, transform 0.3s ease;
  will-change: width, opacity, transform;
}

.panel-wrapper.left {
  border-right: 1px solid #EAEAEA;
}
</style>
