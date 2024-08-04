<!-- YOU CAN DELETE EVERYTHING IN THIS PAGE -->

<div class="container mx-auto flex h-full items-center justify-center">
  <div class="flex flex-col items-center space-y-10 text-center">
    <h1 class="h1">Shayan Amani</h1>
    <h3 class="h3">a.k.a. Shawn Azdam</h3>
    <h5 class="h5">a.k.a. Shawyan</h5>
    <h7 class="h7">a.k.a. SHi-ON</h7>
    <!-- Animated Logo -->
    <figure>
      <section class="img-bg" />
      <img class="fill-token" src="/SHi-ON-logo.svg" alt="SHi ON Logo" />
    </figure>
    <!-- / -->
    <!--    <div class="flex justify-center space-x-2">-->
    <!--      <a-->
    <!--        class="variant-filled btn"-->
    <!--        href="https://skeleton.dev/"-->
    <!--        target="_blank"-->
    <!--        rel="noreferrer">-->
    <!--        Launch Documentation-->
    <!--      </a>-->
    <!--    </div>-->
    <!--    <div class="space-y-2">-->
    <!--      <p>Try editing the following:</p>-->
    <!--      <p><code class="code">/src/routes/+layout.svelte</code></p>-->
    <!--      <p><code class="code">/src/routes/+page.svelte</code></p>-->
    <!--    </div>-->
  </div>
</div>

<style lang="postcss">
  figure {
    @apply relative flex flex-col;
  }

  figure img,
  .img-bg {
    @apply h-64 w-64 md:h-80 md:w-80;
  }

  .img-bg {
    @apply absolute z-[-1] rounded-full blur-[50px] transition-all;
    animation:
      pulse 5s cubic-bezier(0, 0, 0, 0.5) infinite,
      glow 5s linear infinite;
  }

  @keyframes glow {
    0% {
      @apply bg-primary-400/50;
    }
    33% {
      @apply bg-secondary-400/50;
    }
    66% {
      @apply bg-tertiary-400/50;
    }
    100% {
      @apply bg-primary-400/50;
    }
  }

  @keyframes pulse {
    50% {
      transform: scale(2.5);
    }
  }
</style>
