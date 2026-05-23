import { ref } from 'vue'
import { getRunOptions } from '../store/pendingUpload'

const advancedStages = ref({
  createSimulation: false,
  startSimulation: false,
  generateReport: false
})

export function usePipelineAutopilot() {
  const options = getRunOptions()
  const isAutopilotEnabled = () => options.autopilot === true

  const shouldAdvance = (stage) => {
    return isAutopilotEnabled() && !advancedStages.value[stage]
  }

  const markAdvanced = (stage) => {
    advancedStages.value[stage] = true
  }

  return {
    options,
    isAutopilotEnabled,
    shouldAdvance,
    markAdvanced
  }
}
