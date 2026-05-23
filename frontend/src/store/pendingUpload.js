/**
 * Holds pending files and simulation text after "Start" on Home;
 * Process.vue performs the ontology API once mounted.
 */
import { reactive } from 'vue'

const DEFAULT_RUN_OPTIONS = {
  autopilot: true,
  maxRounds: null,              // null => auto rounds (do not send max_rounds)
  useVertexSearch: true,
  enableTwitter: true,
  enableReddit: true,
  enableGraphMemoryUpdate: true,
  parallelProfileCount: 20,
  useLlmForProfiles: true,
}

const state = reactive({
  files: [],
  simulationRequirement: '',
  useVertexSearch: false,
  isPending: false,
  runOptions: { ...DEFAULT_RUN_OPTIONS }
})

export function setPendingUpload(files, requirement, useVertexSearch = false, runOptions = {}) {
  state.files = files
  state.simulationRequirement = requirement
  const mergedRunOptions = { ...DEFAULT_RUN_OPTIONS, ...runOptions }
  state.runOptions = mergedRunOptions
  state.useVertexSearch = Boolean(mergedRunOptions.useVertexSearch ?? useVertexSearch)
  state.isPending = true
  
  // Persist run options for page refresh survival (only these, not files)
  try {
    sessionStorage.setItem('mirofish:runOptions', JSON.stringify(state.runOptions))
  } catch (err) {
    console.warn('Failed to save run options to sessionStorage', err)
  }
}

export function getPendingUpload() {
  return {
    files: state.files,
    simulationRequirement: state.simulationRequirement,
    useVertexSearch: state.useVertexSearch,
    isPending: state.isPending,
    runOptions: { ...state.runOptions }
  }
}

export function getRunOptions() {
  try {
    const stored = sessionStorage.getItem('mirofish:runOptions')
    if (stored) {
      return { ...DEFAULT_RUN_OPTIONS, ...JSON.parse(stored) }
    }
  } catch (err) {
    console.warn('Failed to read run options from sessionStorage', err)
  }
  return { ...state.runOptions }
}

export function clearPendingUpload() {
  state.files = []
  state.simulationRequirement = ''
  state.useVertexSearch = false
  state.runOptions = { ...DEFAULT_RUN_OPTIONS }
  state.isPending = false
}

export default state
