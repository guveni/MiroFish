/**
 * Holds pending files and simulation text after "Start" on Home;
 * Process.vue performs the ontology API once mounted.
 */
import { reactive } from 'vue'

const state = reactive({
  files: [],
  simulationRequirement: '',
  useVertexSearch: false,
  isPending: false
})

export function setPendingUpload(files, requirement, useVertexSearch = false) {
  state.files = files
  state.simulationRequirement = requirement
  state.useVertexSearch = Boolean(useVertexSearch)
  state.isPending = true
}

export function getPendingUpload() {
  return {
    files: state.files,
    simulationRequirement: state.simulationRequirement,
    useVertexSearch: state.useVertexSearch,
    isPending: state.isPending
  }
}

export function clearPendingUpload() {
  state.files = []
  state.simulationRequirement = ''
  state.useVertexSearch = false
  state.isPending = false
}

export default state
