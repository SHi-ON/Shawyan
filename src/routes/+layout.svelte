<script lang="ts">
  import '../app.postcss';
  import { AppShell, AppBar } from '@skeletonlabs/skeleton';
</script>

<!-- App Shell -->
<AppShell>
  <svelte:fragment slot="header">
    <!-- App Bar -->
    <AppBar>
      <svelte:fragment slot="lead">
        <strong class="text-xl">Infinite Eternity</strong>
      </svelte:fragment>
      <svelte:fragment slot="trail">
        <a
          class="variant-ghost-primary btn"
          href="https://x.shawyan.com/github"
          target="_blank"
          rel="noreferrer">
          GitHub
        </a>
        <a
          class="variant-ghost-primary btn"
          href="https://x.shawyan.com/google-scholar"
          target="_blank"
          rel="noreferrer">
          Google Scholar
        </a>
        <a
          class="variant-ghost-primary btn"
          href="https://x.shawyan.com/linkedin"
          target="_blank"
          rel="noreferrer">
          LinkedIn
        </a>
        <a
          class="variant-ghost-primary btn"
          href="https://x.shawyan.com/x"
          target="_blank"
          rel="noreferrer">
          X
        </a>
        <a
          class="variant-ghost-primary btn"
          href="https://x.shawyan.com/stack-overflow"
          target="_blank"
          rel="noreferrer">
          Stack Overflow
        </a>
      </svelte:fragment>
    </AppBar>
  </svelte:fragment>
  <!-- Page Route Content -->
  <slot />
</AppShell>
