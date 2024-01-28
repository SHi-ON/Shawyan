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
          href="https://github.com/SHi-ON"
          target="_blank"
          rel="noreferrer">
          GitHub
        </a>
        <a
          class="variant-ghost-primary btn"
          href="https://scholar.google.com/citations?user=E_sbK9sAAAAJ&hl=en"
          target="_blank"
          rel="noreferrer">
          Google Scholar
        </a>
        <a
          class="variant-ghost-primary btn"
          href="https://www.linkedin.com/in/shayanamani/"
          target="_blank"
          rel="noreferrer">
          LinkedIn
        </a>
        <a
          class="variant-ghost-primary btn"
          href="https://stackoverflow.com/users/3782119/shawyan-azdam"
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
