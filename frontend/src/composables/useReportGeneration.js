import { ref, nextTick } from 'vue'
import { getWebSocketUrl, generateReport, getReport, getAgentLog, getReportSections } from '../api/report'
import { usePipelineAutopilot } from './usePipelineAutopilot'

export function useReportGeneration({
  reportId,
  simulationId,
  agentLogs,
  consoleLogs,
  agentLogLine,
  consoleLogLine,
  onStatusUpdate,
  addLog,
  rightPanelRef,
  logContentRef,
}) {
  const { options: runOptions } = usePipelineAutopilot()

  const reportOutline = ref(null)
  const currentSectionIndex = ref(null)
  const generatedSections = ref({})
  const expandedContent = ref(new Set())
  const collapsedSections = ref(new Set())
  const isComplete = ref(false)
  const isFailed = ref(false)
  const isContinuing = ref(false)
  const isRecreating = ref(false)
  const startTime = ref(null)

  let agentLogWs = null
  let consoleLogWs = null

  const applyAgentLogs = (newLogs) => {
    if (!newLogs?.length) return

    newLogs.forEach(log => {
      agentLogs.value.push(log)

      if (log.action === 'planning_complete' && log.details?.outline) {
        reportOutline.value = log.details.outline
      }

      if (log.action === 'section_start') {
        currentSectionIndex.value = log.section_index
      }

      if (log.action === 'section_complete') {
        if (log.details?.content) {
          generatedSections.value[log.section_index] = log.details.content
          expandedContent.value.add(log.section_index - 1)
          currentSectionIndex.value = null
        }
      }

      if (log.action === 'report_complete') {
        isComplete.value = true
        isFailed.value = false
        currentSectionIndex.value = null
        if (onStatusUpdate) onStatusUpdate('completed')
        stopPolling()

        if (runOptions.autopilot && addLog) {
          addLog('Autopilot: Pipeline complete.')
        }
      }

      if (log.action === 'error') {
        isFailed.value = true
        currentSectionIndex.value = null
        if (onStatusUpdate) onStatusUpdate('error')
        stopPolling()
      }

      if (log.action === 'report_start') {
        startTime.value = new Date(log.timestamp)
      }
    })

    nextTick(() => {
      if (rightPanelRef?.value) {
        if (isComplete.value) {
          rightPanelRef.value.scrollTop = 0
        } else {
          rightPanelRef.value.scrollTop = rightPanelRef.value.scrollHeight
        }
      }
    })
  }

  const loadExistingReport = async () => {
    const id = typeof reportId === 'function' ? reportId() : reportId
    if (!id) return false

    try {
      const reportRes = await getReport(id)
      if (reportRes?.success && reportRes.data) {
        const data = reportRes.data
        if (data.outline?.sections?.length) {
          reportOutline.value = data.outline
        }
        if (data.status === 'completed') {
          isComplete.value = true
          isFailed.value = false
          if (onStatusUpdate) onStatusUpdate('completed')
        }
      }

      const sectionsRes = await getReportSections(id)
      if (sectionsRes?.success && sectionsRes.data?.sections?.length) {
        sectionsRes.data.sections.forEach((section) => {
          if (section.content && section.section_index) {
            generatedSections.value[section.section_index] = section.content
            expandedContent.value.add(section.section_index - 1)
          }
        })
        if (sectionsRes.data.is_complete) {
          isComplete.value = true
          isFailed.value = false
          if (onStatusUpdate) onStatusUpdate('completed')
        }
      }

      const logRes = await getAgentLog(id, 0)
      if (logRes?.success && logRes.data?.logs?.length) {
        applyAgentLogs(logRes.data.logs)
        agentLogLine.value = logRes.data.total_lines || logRes.data.logs.length
      }

      return Boolean(reportOutline.value) || isComplete.value
    } catch (err) {
      console.warn('Could not hydrate existing report state:', err)
      return false
    }
  }

  const toggleSectionContent = (idx) => {
    if (!generatedSections.value[idx + 1]) return
    const newSet = new Set(expandedContent.value)
    if (newSet.has(idx)) {
      newSet.delete(idx)
    } else {
      newSet.add(idx)
    }
    expandedContent.value = newSet
  }

  const toggleSectionCollapse = (idx) => {
    if (!generatedSections.value[idx + 1]) return
    const newSet = new Set(collapsedSections.value)
    if (newSet.has(idx)) {
      newSet.delete(idx)
    } else {
      newSet.add(idx)
    }
    collapsedSections.value = newSet
  }

  const stopPolling = () => {
    if (agentLogWs) {
      agentLogWs.close()
      agentLogWs = null
    }
    if (consoleLogWs) {
      consoleLogWs.close()
      consoleLogWs = null
    }
  }

  const startPolling = async () => {
    if (agentLogWs || consoleLogWs) return
    const id = typeof reportId === 'function' ? reportId() : reportId
    if (!id) return

    const hydrated = await loadExistingReport()
    if (hydrated && isComplete.value) {
      return
    }
    
    const agentUrl = getWebSocketUrl(`/api/report/${id}/agent-log/ws`)
    const consoleUrl = getWebSocketUrl(`/api/report/${id}/console-log/ws`)
    
    agentLogWs = new WebSocket(agentUrl)
    agentLogWs.onmessage = (event) => {
      try {
        const res = JSON.parse(event.data)
        if (res.success && res.data) {
          const newLogs = res.data.logs || []
          if (newLogs.length > 0) {
            applyAgentLogs(newLogs)
            agentLogLine.value = res.data.from_line + newLogs.length
          }
        }
      } catch (err) {
        console.error('Error processing agent log ws message:', err)
      }
    }
    agentLogWs.onerror = (err) => {
      console.warn('Agent log WebSocket error:', err)
    }
    agentLogWs.onclose = () => {
      agentLogWs = null
    }
    
    consoleLogWs = new WebSocket(consoleUrl)
    consoleLogWs.onmessage = (event) => {
      try {
        const res = JSON.parse(event.data)
        if (res.success && res.data) {
          const newLogs = res.data.logs || []
          if (newLogs.length > 0) {
            consoleLogs.value.push(...newLogs)
            consoleLogLine.value = res.data.from_line + newLogs.length
            
            nextTick(() => {
              if (logContentRef?.value) {
                logContentRef.value.scrollTop = logContentRef.value.scrollHeight
              }
            })
          }
        }
      } catch (err) {
        console.error('Error processing console log ws message:', err)
      }
    }
    consoleLogWs.onerror = (err) => {
      console.warn('Console log WebSocket error:', err)
    }
    consoleLogWs.onclose = () => {
      consoleLogWs = null
    }
  }

  const handleContinue = async () => {
    if (isContinuing.value) return
    isContinuing.value = true
    isFailed.value = false
    if (onStatusUpdate) onStatusUpdate('processing')
    
    const rId = typeof reportId === 'function' ? reportId() : reportId
    const sId = typeof simulationId === 'function' ? simulationId() : simulationId

    try {
      const res = await generateReport({
        simulation_id: sId,
        report_id: rId,
        force_regenerate: false
      })
      
      if (res.success) {
        startPolling()
      } else {
        isFailed.value = true
        if (onStatusUpdate) onStatusUpdate('error')
      }
    } catch (err) {
      console.error('Continue report failed:', err)
      isFailed.value = true
      if (onStatusUpdate) onStatusUpdate('error')
    } finally {
      isContinuing.value = false
    }
  }

  return {
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
  }
}
