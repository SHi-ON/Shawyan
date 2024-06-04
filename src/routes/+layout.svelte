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
          class="variant-glass-primary btn"
          href="https://x.shawyan.com/github"
          target="_blank"
          rel="noreferrer">
          <img
            height="28"
            width="28"
            alt="GitHub icon"
            src="https://cdn.simpleicons.org/github/white" />
        </a>
        <a
          class="variant-glass-primary btn"
          href="https://x.shawyan.com/google-scholar"
          target="_blank"
          rel="noreferrer">
          <img
            height="28"
            width="28"
            alt="Google Scholar icon"
            src="https://cdn.simpleicons.org/googlescholar/white" />
        </a>
        <a
          class="variant-glass-primary btn"
          href="https://x.shawyan.com/linkedin"
          target="_blank"
          rel="noreferrer">
          <img
            height="28"
            width="28"
            alt="LinkedIn icon"
            src="https://cdn.simpleicons.org/linkedin/white" />
        </a>
        <a
          class="variant-glass-primary btn"
          href="https://x.shawyan.com/x"
          target="_blank"
          rel="noreferrer">
          <img
            height="28"
            width="28"
            alt="X icon"
            src="https://cdn.simpleicons.org/x/white" />
        </a>
        <a
          class="variant-glass-primary btn"
          href="https://x.shawyan.com/stack-overflow"
          target="_blank"
          rel="noreferrer">
          <img
            height="28"
            width="28"
            alt="Stack Overflow icon"
            src="https://cdn.simpleicons.org/stackoverflow/white" />
        </a>
      </svelte:fragment>
    </AppBar>
  </svelte:fragment>
  <!-- Page Route Content -->
  <slot />
</AppShell>
