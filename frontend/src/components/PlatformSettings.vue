<template>
  <div class="config-block">
    <div class="config-block-header">
      <span class="config-block-title">{{ $t('step2.recommendAlgoConfig') }}</span>
    </div>
    <div class="platforms-grid">
      <div v-if="twitterConfig" class="platform-card">
        <div class="platform-card-header">
          <span class="platform-name">{{ $t('step2.platform1Name') }}</span>
        </div>
        <div class="platform-params">
          <div class="param-row">
            <span class="param-label">{{ $t('step2.recencyWeight') }}</span>
            <span class="param-value">{{ twitterConfig.recency_weight }}</span>
          </div>
          <div class="param-row">
            <span class="param-label">{{ $t('step2.popularityWeight') }}</span>
            <span class="param-value">{{ twitterConfig.popularity_weight }}</span>
          </div>
          <div class="param-row">
            <span class="param-label">{{ $t('step2.relevanceWeight') }}</span>
            <span class="param-value">{{ twitterConfig.relevance_weight }}</span>
          </div>
          <div class="param-row">
            <span class="param-label">{{ $t('step2.viralThreshold') }}</span>
            <span class="param-value">{{ twitterConfig.viral_threshold }}</span>
          </div>
          <div class="param-row">
            <span class="param-label">{{ $t('step2.echoChamberStrength') }}</span>
            <span class="param-value">{{ twitterConfig.echo_chamber_strength }}</span>
          </div>
        </div>
      </div>
      <div v-if="redditConfig" class="platform-card">
        <div class="platform-card-header">
          <span class="platform-name">{{ $t('step2.platform2Name') }}</span>
        </div>
        <div class="platform-params">
          <div class="param-row">
            <span class="param-label">{{ $t('step2.recencyWeight') }}</span>
            <span class="param-value">{{ redditConfig.recency_weight }}</span>
          </div>
          <div class="param-row">
            <span class="param-label">{{ $t('step2.popularityWeight') }}</span>
            <span class="param-value">{{ redditConfig.popularity_weight }}</span>
          </div>
          <div class="param-row">
            <span class="param-label">{{ $t('step2.relevanceWeight') }}</span>
            <span class="param-value">{{ redditConfig.relevance_weight }}</span>
          </div>
          <div class="param-row">
            <span class="param-label">{{ $t('step2.viralThreshold') }}</span>
            <span class="param-value">{{ redditConfig.viral_threshold }}</span>
          </div>
          <div class="param-row">
            <span class="param-label">{{ $t('step2.echoChamberStrength') }}</span>
            <span class="param-value">{{ redditConfig.echo_chamber_strength }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  twitterConfig: Object,
  redditConfig: Object
})
</script>
